#include <torch/extension.h>

#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>

#include <algorithm>
#include <dlfcn.h>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace fs = std::filesystem;

namespace {

struct RmsNormParams {
  uint32_t rows;
  uint32_t dim;
  float eps;
  uint32_t x_stride;
  uint32_t out_stride;
  uint32_t has_weight;
  uint32_t has_residual;
};

struct AdaLNParams {
  uint32_t rows;
  uint32_t dim;
  float eps;
  uint32_t x_stride;
  uint32_t scale_stride;
  uint32_t shift_stride;
  uint32_t out_stride;
  uint32_t use_rms;
};

struct RopePairParams {
  uint32_t B;
  uint32_t H;
  uint32_t T;
  uint32_t pairs;
  uint32_t x_stride_b;
  uint32_t x_stride_h;
  uint32_t x_stride_t;
  uint32_t f_stride_b;
  uint32_t f_stride_h;
  uint32_t f_stride_t;
  uint32_t f_stride_pair;
};

struct RopeSplitParams {
  uint32_t B;
  uint32_t H;
  uint32_t T;
  uint32_t half_d;
  uint32_t x_stride_b;
  uint32_t x_stride_h;
  uint32_t x_stride_t;
  uint32_t f_stride_b;
  uint32_t f_stride_h;
  uint32_t f_stride_t;
  uint32_t f_stride_pair;
};

struct QuantRowParams {
  uint32_t rows;
  uint32_t dim;
  uint32_t x_stride;
  uint32_t q_stride;
  uint32_t stochastic;
  uint32_t seed_lo;
  uint32_t seed_hi;
  uint32_t _pad;
};

inline void set_quant_seed(QuantRowParams& p, bool stochastic, int64_t seed) {
  p.stochastic = stochastic ? 1u : 0u;
  const uint64_t s = static_cast<uint64_t>(seed);
  p.seed_lo = static_cast<uint32_t>(s);
  p.seed_hi = static_cast<uint32_t>(s >> 32);
  p._pad = 0;
}

struct RmsRopeParams {
  uint32_t B;
  uint32_t H;
  uint32_t T;
  uint32_t D;
  float eps;
  uint32_t x_stride_b;
  uint32_t x_stride_h;
  uint32_t x_stride_t;
  uint32_t out_stride_b;
  uint32_t out_stride_h;
  uint32_t out_stride_t;
  uint32_t f_stride_b;
  uint32_t f_stride_h;
  uint32_t f_stride_t;
  uint32_t f_stride_pair;
  uint32_t rot_dim;
  uint32_t split_half;
  uint32_t bnhd;
};

struct RmsInvParams {
  uint32_t rows;
  uint32_t dim;
  float eps;
  uint32_t x_stride;
};

struct Fp8Params {
  uint32_t numel;
  uint32_t fmt;
};

struct TensorQuantParams {
  uint32_t numel;
  uint32_t stochastic;
  uint32_t seed_lo;
  uint32_t seed_hi;
};

inline void set_tensor_quant_seed(TensorQuantParams& p, bool stochastic, int64_t seed) {
  p.stochastic = stochastic ? 1u : 0u;
  const uint64_t s = static_cast<uint64_t>(seed);
  p.seed_lo = static_cast<uint32_t>(s);
  p.seed_hi = static_cast<uint32_t>(s >> 32);
}

struct DequantFlatParams {
  uint32_t numel;
  uint32_t inner_dim;
  uint32_t scale_mode;
};

struct SvdDequantParams {
  uint32_t rows;
  uint32_t k;
  uint32_t groups;
  uint32_t packed_stride;
  uint32_t scale_stride;
  uint32_t out_stride;
  uint32_t unsigned_act;
};

struct SvdQuantParams {
  uint32_t rows;
  uint32_t k;
  uint32_t groups;
  uint32_t x_stride;
  uint32_t packed_stride;
  uint32_t scale_stride;
  uint32_t unsigned_act;
  uint32_t qmax;
};

constexpr NSUInteger kReduceThreads = 256;
constexpr NSUInteger kFp8Threads = 256;
constexpr uint32_t kSvdGroupSize = 64;

static id<MTLLibrary> g_lib = nil;
static std::mutex g_mutex;
static std::unordered_map<std::string, id<MTLComputePipelineState>> g_pipelines;

static std::string metallib_path_for_this_module() {
  Dl_info info;
  if (dladdr((void*)&metallib_path_for_this_module, &info) == 0 || info.dli_fname == nullptr) {
    return std::string();
  }
  return (fs::path(info.dli_fname).parent_path() / "comfy_kitchen_metal.metallib").string();
}

static void ensure_library_locked(id<MTLDevice> device) {
  if (g_lib != nil) {
    return;
  }
  std::string lib_path = metallib_path_for_this_module();
  if (const char* env = getenv("COMFY_KITCHEN_METALLIB")) {
    if (env[0] != '\0') {
      lib_path = env;
    }
  }
  TORCH_CHECK(!lib_path.empty(), "comfy_kitchen metal: metallib path not found");
  NSError* err = nil;
  g_lib = [device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:lib_path.c_str()]]
                             error:&err];
  if (g_lib == nil) {
    const char* msg = err ? [[err localizedDescription] UTF8String] : "unknown";
    TORCH_CHECK(false, "comfy_kitchen metal: failed to load metallib: ", msg);
  }
}

static id<MTLComputePipelineState> get_pipeline(id<MTLDevice> device, const char* name) {
  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_library_locked(device);
  std::string key(name);
  auto it = g_pipelines.find(key);
  if (it != g_pipelines.end()) {
    return it->second;
  }
  id<MTLFunction> fn = [g_lib newFunctionWithName:[NSString stringWithUTF8String:name]];
  TORCH_CHECK(fn != nil, "comfy_kitchen metal: kernel missing: ", name);
  NSError* err = nil;
  id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
  TORCH_CHECK(pso != nil, "comfy_kitchen metal: pipeline failed for ", name);
  g_pipelines.emplace(key, pso);
  return pso;
}

static inline id<MTLBuffer> storage_as_mtlbuffer(const at::Tensor& t) {
  void* ctx = t.storage().data_ptr().get_context();
  TORCH_CHECK(ctx != nullptr, "comfy_kitchen metal: expected MPS MTLBuffer context");
  return (__bridge id<MTLBuffer>)ctx;
}

static inline NSUInteger storage_offset_bytes(const at::Tensor& t) {
  return (NSUInteger)(t.storage_offset() * (int64_t)t.element_size());
}

static uint32_t stride_u32(const at::Tensor& t, int idx, const char* name) {
  const int64_t s = t.stride(idx);
  TORCH_CHECK(s >= 0 && s <= (int64_t)std::numeric_limits<uint32_t>::max(), name, " stride");
  return (uint32_t)s;
}

static void check_mps_fp(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_mps(), name, ": must be MPS");
  TORCH_CHECK(t.dtype() == at::kHalf || t.dtype() == at::kBFloat16, name, ": fp16/bf16");
}

static const char* suffix(c10::ScalarType dtype) {
  return dtype == at::kBFloat16 ? "bf16" : "fp16";
}

static void set_tensor(id<MTLComputeCommandEncoder> enc, const at::Tensor& t, int idx) {
  [enc setBuffer:storage_as_mtlbuffer(t) offset:storage_offset_bytes(t) atIndex:(NSUInteger)idx];
}

at::Tensor residual_rms_norm(
    at::Tensor x,
    c10::optional<at::Tensor> weight_opt,
    c10::optional<at::Tensor> residual_opt,
    double eps) {
  check_mps_fp(x, "residual_rms_norm");
  TORCH_CHECK(x.stride(-1) == 1, "last dim contiguous");
  const int64_t dim = x.size(-1);
  auto x2 = x.reshape({-1, dim});
  auto out = at::empty_like(x2);

  at::Tensor weight = at::empty({0}, x.options());
  uint32_t has_weight = 0;
  if (weight_opt.has_value() && weight_opt->defined()) {
    weight = weight_opt->contiguous();
    check_mps_fp(weight, "weight");
    TORCH_CHECK(weight.numel() == dim);
    has_weight = 1;
  }
  at::Tensor residual = at::empty({0}, x.options());
  uint32_t has_residual = 0;
  if (residual_opt.has_value() && residual_opt->defined()) {
    residual = residual_opt->reshape({-1, dim});
    check_mps_fp(residual, "residual");
    has_residual = 1;
  }

  RmsNormParams p;
  p.rows = (uint32_t)x2.size(0);
  p.dim = (uint32_t)dim;
  p.eps = (float)eps;
  p.x_stride = stride_u32(x2, 0, "x");
  p.out_stride = stride_u32(out, 0, "out");
  p.has_weight = has_weight;
  p.has_residual = has_residual;

  std::string kname = std::string("residual_rms_norm_") + suffix(x.scalar_type());
  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = get_pipeline(device, kname.c_str());
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [enc setComputePipelineState:pipeline];
    set_tensor(enc, x2, 0);
    set_tensor(enc, has_weight ? weight : x2, 1);
    set_tensor(enc, has_residual ? residual : x2, 2);
    set_tensor(enc, out, 3);
    [enc setBytes:&p length:sizeof(RmsNormParams) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(p.rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kReduceThreads, 1, 1)];
    stream->endKernelCoalescing();
  });
  return out.reshape_as(x);
}

at::Tensor rms_gated_residual(
    at::Tensor activation,
    at::Tensor norm_weight,
    at::Tensor residual,
    at::Tensor gate,
    double eps) {
  check_mps_fp(activation, "rms_gated_residual:activation");
  check_mps_fp(norm_weight, "rms_gated_residual:norm_weight");
  check_mps_fp(residual, "rms_gated_residual:residual");
  check_mps_fp(gate, "rms_gated_residual:gate");
  TORCH_CHECK(activation.sizes() == residual.sizes());
  TORCH_CHECK(activation.stride(-1) == 1 && residual.stride(-1) == 1);
  const int64_t dim = activation.size(-1);
  TORCH_CHECK(norm_weight.numel() == dim && gate.numel() == dim);
  auto x2 = activation.reshape({-1, dim}).contiguous();
  auto r2 = residual.reshape({-1, dim}).contiguous();
  auto w = norm_weight.reshape({dim}).contiguous();
  auto g = gate.reshape({dim}).contiguous();
  auto out = at::empty_like(x2);

  RmsNormParams p;
  p.rows = (uint32_t)x2.size(0);
  p.dim = (uint32_t)dim;
  p.eps = (float)eps;
  p.x_stride = stride_u32(x2, 0, "x");
  p.out_stride = stride_u32(out, 0, "out");
  p.has_weight = 1u;
  p.has_residual = 1u;

  std::string kname = std::string("rms_gated_residual_") + suffix(activation.scalar_type());
  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = get_pipeline(device, kname.c_str());
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [enc setComputePipelineState:pipeline];
    set_tensor(enc, x2, 0);
    set_tensor(enc, w, 1);
    set_tensor(enc, r2, 2);
    set_tensor(enc, g, 3);
    set_tensor(enc, out, 4);
    [enc setBytes:&p length:sizeof(RmsNormParams) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(p.rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kReduceThreads, 1, 1)];
    stream->endKernelCoalescing();
  });
  return out.reshape_as(activation);
}

at::Tensor adaln(at::Tensor x, at::Tensor scale, at::Tensor shift, double eps, bool use_rms) {
  check_mps_fp(x, "adaln:x");
  check_mps_fp(scale, "adaln:scale");
  check_mps_fp(shift, "adaln:shift");
  TORCH_CHECK(x.stride(-1) == 1, "x last dim contiguous");
  const int64_t dim = x.size(-1);
  auto x2 = x.reshape({-1, dim});
  auto out = at::empty_like(x2);

  auto scale2 = scale.reshape({-1, dim});
  auto shift2 = shift.reshape({-1, dim});
  TORCH_CHECK(scale2.size(-1) == dim && shift2.size(-1) == dim);
  const bool scale_bcast = scale2.size(0) == 1;
  const bool shift_bcast = shift2.size(0) == 1;
  TORCH_CHECK(scale_bcast || scale2.size(0) == x2.size(0));
  TORCH_CHECK(shift_bcast || shift2.size(0) == x2.size(0));

  AdaLNParams p;
  p.rows = (uint32_t)x2.size(0);
  p.dim = (uint32_t)dim;
  p.eps = (float)eps;
  p.x_stride = stride_u32(x2, 0, "x");
  p.scale_stride = scale_bcast ? 0u : stride_u32(scale2, 0, "scale");
  p.shift_stride = shift_bcast ? 0u : stride_u32(shift2, 0, "shift");
  p.out_stride = stride_u32(out, 0, "out");
  p.use_rms = use_rms ? 1u : 0u;

  std::string kname = std::string("adaln_") + suffix(x.scalar_type());
  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = get_pipeline(device, kname.c_str());
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [enc setComputePipelineState:pipeline];
    set_tensor(enc, x2, 0);
    set_tensor(enc, scale2, 1);
    set_tensor(enc, shift2, 2);
    set_tensor(enc, out, 3);
    [enc setBytes:&p length:sizeof(AdaLNParams) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(p.rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kReduceThreads, 1, 1)];
    stream->endKernelCoalescing();
  });
  return out.reshape_as(x);
}

struct RopeLayout {
  int64_t B = 0;
  int64_t H = 0;
  int64_t T = 0;
  int64_t D = 0;
  bool bnhd = false;
};

// Detect BHND vs BNHD from freqs vs x without copying x.
static RopeLayout detect_rope_layout(const at::Tensor& x, const at::Tensor& freqs) {
  TORCH_CHECK(x.dim() == 4 && x.size(-1) % 2 == 0);
  RopeLayout L;
  L.B = x.size(0);
  L.D = x.size(3);
  L.H = x.size(1);
  L.T = x.size(2);
  L.bnhd = false;
  if (freqs.dim() != 6) {
    return L;
  }
  const int64_t f1 = freqs.size(1);
  const int64_t f2 = freqs.size(2);
  // (B, 1, T, ...) BHND broadcast
  if (f1 == 1 && (f2 == 1 || f2 == x.size(2))) {
    L.bnhd = false;
    L.H = x.size(1);
    L.T = x.size(2);
    return L;
  }
  // (B, T, 1, ...) BNHD broadcast
  if (f2 == 1 && (f1 == 1 || f1 == x.size(1))) {
    L.bnhd = true;
    L.T = x.size(1);
    L.H = x.size(2);
    return L;
  }
  // Per-head freqs aligned with x's middle dims.
  if (f1 == x.size(1) && f2 == x.size(2)) {
    const int64_t d1 = x.size(1);
    const int64_t d2 = x.size(2);
    if (d1 <= 256 && d2 > d1) {
      L.bnhd = false;
      L.H = d1;
      L.T = d2;
    } else if (d2 <= 256 && d1 > d2) {
      L.bnhd = true;
      L.T = d1;
      L.H = d2;
    } else {
      // Tie-break: LTX freqs_cis_matrix is (B, T, H, ...).
      L.bnhd = true;
      L.T = d1;
      L.H = d2;
    }
    return L;
  }
  return L;
}

struct PreparedFreqs {
  at::Tensor storage;
  uint32_t stride_b = 0;
  uint32_t stride_h = 0;
  uint32_t stride_t = 0;
  uint32_t stride_pair = 0;
};

// Prepare freqs for logical index f[b, h, t, pair] without requiring a singleton
// head axis. Broadcast heads use stride_h = 0. No permute of x.
static PreparedFreqs prepare_freqs(
    at::Tensor freqs,
    int64_t B,
    int64_t H,
    int64_t T,
    int64_t pairs,
    bool x_bnhd) {
  TORCH_CHECK(freqs.device().is_mps());
  TORCH_CHECK(freqs.size(-1) == 2 && freqs.size(-2) == 2);
  if (freqs.dtype() != at::kFloat) {
    freqs = freqs.to(at::kFloat);
  }

  PreparedFreqs out;
  if (freqs.dim() == 6) {
    const int64_t f1 = freqs.size(1);
    const int64_t f2 = freqs.size(2);
    const int64_t fp = freqs.size(3);

    if (f1 == 1 && (f2 == T || f2 == 1)) {
      // BHND-style (B', 1, T', pairs, 2, 2)
      if (freqs.size(0) == 1 && B > 1) {
        freqs = freqs.expand({B, 1, freqs.size(2), fp, 2, 2});
      }
      if (freqs.size(2) == 1 && T > 1) {
        freqs = freqs.expand({freqs.size(0), 1, T, fp, 2, 2});
      }
      if (fp == 1 && pairs > 1) {
        freqs = freqs.expand({freqs.size(0), 1, freqs.size(2), pairs, 2, 2});
      }
      freqs = freqs.contiguous();
      TORCH_CHECK(freqs.size(0) == B && freqs.size(2) == T && freqs.size(3) == pairs);
      out.storage = freqs;
      out.stride_b = (uint32_t)freqs.stride(0);
      out.stride_h = 0;
      out.stride_t = (uint32_t)freqs.stride(2);
      out.stride_pair = (uint32_t)freqs.stride(3);
      return out;
    }

    if (f2 == 1 && (f1 == T || f1 == 1)) {
      // BNHD-style (B', T', 1, pairs, 2, 2)
      if (freqs.size(0) == 1 && B > 1) {
        freqs = freqs.expand({B, freqs.size(1), 1, fp, 2, 2});
      }
      if (freqs.size(1) == 1 && T > 1) {
        freqs = freqs.expand({freqs.size(0), T, 1, fp, 2, 2});
      }
      if (fp == 1 && pairs > 1) {
        freqs = freqs.expand({freqs.size(0), freqs.size(1), 1, pairs, 2, 2});
      }
      freqs = freqs.contiguous();
      TORCH_CHECK(freqs.size(0) == B && freqs.size(1) == T && freqs.size(3) == pairs);
      out.storage = freqs;
      out.stride_b = (uint32_t)freqs.stride(0);
      out.stride_h = 0;
      out.stride_t = (uint32_t)freqs.stride(1);
      out.stride_pair = (uint32_t)freqs.stride(3);
      return out;
    }

    if (x_bnhd && f1 == T && f2 == H) {
      // Per-head BNHD (B, T, H, pairs, 2, 2) — zero-copy strides.
      if (freqs.size(0) == 1 && B > 1) {
        freqs = freqs.expand({B, T, H, fp, 2, 2});
      }
      if (fp == 1 && pairs > 1) {
        freqs = freqs.expand({freqs.size(0), T, H, pairs, 2, 2});
      }
      freqs = freqs.contiguous();
      TORCH_CHECK(
          freqs.size(0) == B && freqs.size(1) == T && freqs.size(2) == H &&
          freqs.size(3) == pairs);
      out.storage = freqs;
      out.stride_b = (uint32_t)freqs.stride(0);
      out.stride_t = (uint32_t)freqs.stride(1);
      out.stride_h = (uint32_t)freqs.stride(2);
      out.stride_pair = (uint32_t)freqs.stride(3);
      return out;
    }

    if (!x_bnhd && f1 == H && f2 == T) {
      // Per-head BHND (B, H, T, pairs, 2, 2)
      if (freqs.size(0) == 1 && B > 1) {
        freqs = freqs.expand({B, H, T, fp, 2, 2});
      }
      if (fp == 1 && pairs > 1) {
        freqs = freqs.expand({freqs.size(0), H, T, pairs, 2, 2});
      }
      freqs = freqs.contiguous();
      TORCH_CHECK(
          freqs.size(0) == B && freqs.size(1) == H && freqs.size(2) == T &&
          freqs.size(3) == pairs);
      out.storage = freqs;
      out.stride_b = (uint32_t)freqs.stride(0);
      out.stride_h = (uint32_t)freqs.stride(1);
      out.stride_t = (uint32_t)freqs.stride(2);
      out.stride_pair = (uint32_t)freqs.stride(3);
      return out;
    }
  }

  // Lower-rank / opaque: treat as (B, T, pairs, 2, 2) with broadcast heads.
  while (freqs.dim() < 5) {
    freqs = freqs.unsqueeze(0);
  }
  if (freqs.dim() == 6) {
    // Ambiguous leftover 6D — squeeze a singleton if present, else fail clearly.
    if (freqs.size(1) == 1) {
      freqs = freqs.squeeze(1);
    } else if (freqs.size(2) == 1) {
      freqs = freqs.squeeze(2);
    } else {
      TORCH_CHECK(
          false,
          "freqs 6D layout does not match x (BHND/BNHD); got ",
          freqs.sizes());
    }
  }
  TORCH_CHECK(freqs.dim() == 5, "freqs must be rank-5, got ", freqs.dim());
  if (freqs.size(-3) == 1 && pairs > 1) {
    freqs = freqs.expand({freqs.size(0), freqs.size(1), pairs, 2, 2});
  }
  if (freqs.size(0) == 1 && B > 1) {
    freqs = freqs.expand({B, freqs.size(1), freqs.size(2), 2, 2});
  }
  if (freqs.size(1) == 1 && T > 1) {
    freqs = freqs.expand({freqs.size(0), T, freqs.size(2), 2, 2});
  }
  freqs = freqs.contiguous();
  TORCH_CHECK(freqs.size(0) == B && freqs.size(1) == T && freqs.size(2) == pairs);
  out.storage = freqs;
  out.stride_b = (uint32_t)freqs.stride(0);
  out.stride_h = 0;
  out.stride_t = (uint32_t)freqs.stride(1);
  out.stride_pair = (uint32_t)freqs.stride(2);
  return out;
}

static void fill_x_strides(const at::Tensor& x, const RopeLayout& L, uint32_t& sb, uint32_t& sh, uint32_t& st) {
  sb = stride_u32(x, 0, "x");
  if (L.bnhd) {
    st = stride_u32(x, 1, "x");
    sh = stride_u32(x, 2, "x");
  } else {
    sh = stride_u32(x, 1, "x");
    st = stride_u32(x, 2, "x");
  }
}

at::Tensor apply_rope1_(at::Tensor x, at::Tensor freqs) {
  check_mps_fp(x, "apply_rope1_");
  TORCH_CHECK(x.dim() == 4 && x.size(-1) % 2 == 0);
  TORCH_CHECK(x.stride(-1) == 1, "apply_rope1_ requires contiguous head_dim");
  const RopeLayout L = detect_rope_layout(x, freqs);
  const int64_t pairs = L.D / 2;
  auto f = prepare_freqs(freqs, L.B, L.H, L.T, pairs, L.bnhd);

  RopePairParams p;
  p.B = (uint32_t)L.B;
  p.H = (uint32_t)L.H;
  p.T = (uint32_t)L.T;
  p.pairs = (uint32_t)pairs;
  fill_x_strides(x, L, p.x_stride_b, p.x_stride_h, p.x_stride_t);
  p.f_stride_b = f.stride_b;
  p.f_stride_h = f.stride_h;
  p.f_stride_t = f.stride_t;
  p.f_stride_pair = f.stride_pair;

  std::string kname = std::string("apply_rope1_inplace_") + suffix(x.scalar_type());
  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = get_pipeline(device, kname.c_str());
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [enc setComputePipelineState:pipeline];
    set_tensor(enc, x, 0);
    set_tensor(enc, f.storage, 1);
    [enc setBytes:&p length:sizeof(RopePairParams) atIndex:2];
    const NSUInteger tg_x = std::min<NSUInteger>((NSUInteger)pairs, 32u);
    [enc dispatchThreads:MTLSizeMake((NSUInteger)pairs, (NSUInteger)L.T, (NSUInteger)(L.B * L.H))
        threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
    stream->endKernelCoalescing();
  });
  return x;
}

at::Tensor apply_rope_split_(at::Tensor x, at::Tensor freqs) {
  check_mps_fp(x, "apply_rope_split_");
  TORCH_CHECK(x.dim() == 4 && x.size(-1) % 2 == 0);
  TORCH_CHECK(x.stride(-1) == 1, "apply_rope_split_ requires contiguous head_dim");
  const RopeLayout L = detect_rope_layout(x, freqs);
  const int64_t half_d = L.D / 2;
  auto f = prepare_freqs(freqs, L.B, L.H, L.T, half_d, L.bnhd);

  RopeSplitParams p;
  p.B = (uint32_t)L.B;
  p.H = (uint32_t)L.H;
  p.T = (uint32_t)L.T;
  p.half_d = (uint32_t)half_d;
  fill_x_strides(x, L, p.x_stride_b, p.x_stride_h, p.x_stride_t);
  p.f_stride_b = f.stride_b;
  p.f_stride_h = f.stride_h;
  p.f_stride_t = f.stride_t;
  p.f_stride_pair = f.stride_pair;

  std::string kname = std::string("apply_rope_split_inplace_") + suffix(x.scalar_type());
  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = get_pipeline(device, kname.c_str());
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [enc setComputePipelineState:pipeline];
    set_tensor(enc, x, 0);
    set_tensor(enc, f.storage, 1);
    [enc setBytes:&p length:sizeof(RopeSplitParams) atIndex:2];
    const NSUInteger tg_x = std::min<NSUInteger>((NSUInteger)half_d, 32u);
    [enc dispatchThreads:MTLSizeMake((NSUInteger)half_d, (NSUInteger)L.T, (NSUInteger)(L.B * L.H))
        threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];
    stream->endKernelCoalescing();
  });
  return x;
}

std::tuple<at::Tensor, at::Tensor> quantize_int8_rowwise(
    at::Tensor x, bool stochastic = false, int64_t seed = 0) {
  check_mps_fp(x, "quantize_int8_rowwise");
  TORCH_CHECK(x.stride(-1) == 1);
  const int64_t dim = x.size(-1);
  auto x2 = x.reshape({-1, dim}).contiguous();
  auto q = at::empty(x2.sizes(), x2.options().dtype(at::kChar));
  auto scale = at::empty({x2.size(0)}, x2.options().dtype(at::kFloat));

  QuantRowParams p{};
  p.rows = (uint32_t)x2.size(0);
  p.dim = (uint32_t)dim;
  p.x_stride = stride_u32(x2, 0, "x");
  p.q_stride = stride_u32(q, 0, "q");
  set_quant_seed(p, stochastic, seed);

  std::string kname = std::string("quantize_int8_rowwise_") + suffix(x.scalar_type());
  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = get_pipeline(device, kname.c_str());
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [enc setComputePipelineState:pipeline];
    set_tensor(enc, x2, 0);
    set_tensor(enc, q, 1);
    set_tensor(enc, scale, 2);
    [enc setBytes:&p length:sizeof(QuantRowParams) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(p.rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kReduceThreads, 1, 1)];
    stream->endKernelCoalescing();
  });

  std::vector<int64_t> sshape(x.sizes().begin(), x.sizes().end());
  sshape.back() = 1;
  return {q.reshape_as(x), scale.reshape(sshape)};
}

std::tuple<at::Tensor, at::Tensor> quantize_int4_rowwise(
    at::Tensor x, bool stochastic = false, int64_t seed = 0) {
  check_mps_fp(x, "quantize_int4_rowwise");
  TORCH_CHECK(x.stride(-1) == 1);
  const int64_t dim = x.size(-1);
  TORCH_CHECK(dim % 2 == 0);
  auto x2 = x.reshape({-1, dim}).contiguous();
  auto packed = at::empty({x2.size(0), dim / 2}, x2.options().dtype(at::kChar));
  auto scale = at::empty({x2.size(0)}, x2.options().dtype(at::kFloat));

  QuantRowParams p{};
  p.rows = (uint32_t)x2.size(0);
  p.dim = (uint32_t)dim;
  p.x_stride = stride_u32(x2, 0, "x");
  p.q_stride = stride_u32(packed, 0, "packed");
  set_quant_seed(p, stochastic, seed);

  std::string kname = std::string("quantize_int4_rowwise_") + suffix(x.scalar_type());
  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = get_pipeline(device, kname.c_str());
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [enc setComputePipelineState:pipeline];
    set_tensor(enc, x2, 0);
    set_tensor(enc, packed, 1);
    set_tensor(enc, scale, 2);
    [enc setBytes:&p length:sizeof(QuantRowParams) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(p.rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kReduceThreads, 1, 1)];
    stream->endKernelCoalescing();
  });
  return {packed, scale};
}

at::Tensor dequantize_int4_rowwise(at::Tensor packed, at::Tensor scale) {
  TORCH_CHECK(packed.device().is_mps() && packed.dtype() == at::kChar);
  TORCH_CHECK(scale.device().is_mps());
  TORCH_CHECK(packed.dim() == 2);
  const int64_t rows = packed.size(0);
  const int64_t pairs = packed.size(1);
  const int64_t dim = pairs * 2;
  auto p2 = packed.contiguous();
  auto scale2 = scale.to(at::kFloat).reshape({-1}).contiguous();
  TORCH_CHECK(scale2.numel() == rows);
  // Emit fp16 when possible; callers cast if they need bf16.
  auto out = at::empty({rows, dim}, packed.options().dtype(at::kHalf));

  QuantRowParams p{};
  p.rows = (uint32_t)rows;
  p.dim = (uint32_t)dim;
  p.x_stride = stride_u32(out, 0, "out");
  p.q_stride = stride_u32(p2, 0, "packed");

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = get_pipeline(device, "dequantize_int4_rowwise_fp16");
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [enc setComputePipelineState:pipeline];
    set_tensor(enc, p2, 0);
    set_tensor(enc, scale2, 1);
    set_tensor(enc, out, 2);
    [enc setBytes:&p length:sizeof(QuantRowParams) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(p.rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kReduceThreads, 1, 1)];
    stream->endKernelCoalescing();
  });
  return out;
}

at::Tensor dequantize_int8_simple(at::Tensor q, at::Tensor scale) {
  TORCH_CHECK(q.device().is_mps() && q.dtype() == at::kChar);
  TORCH_CHECK(scale.device().is_mps());
  const int64_t dim = q.size(-1);
  auto q2 = q.reshape({-1, dim}).contiguous();
  auto out = at::empty(q2.sizes(), q2.options().dtype(at::kFloat));
  auto scale2 = scale.to(at::kFloat).reshape({-1}).contiguous();
  if (scale2.numel() == 1) {
    scale2 = scale2.expand({q2.size(0)}).contiguous();
  }
  TORCH_CHECK(scale2.numel() == q2.size(0));

  QuantRowParams p{};
  p.rows = (uint32_t)q2.size(0);
  p.dim = (uint32_t)dim;
  p.x_stride = stride_u32(out, 0, "out");
  p.q_stride = stride_u32(q2, 0, "q");

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = get_pipeline(device, "dequantize_int8_simple_fp32");
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [enc setComputePipelineState:pipeline];
    set_tensor(enc, q2, 0);
    set_tensor(enc, scale2, 1);
    set_tensor(enc, out, 2);
    [enc setBytes:&p length:sizeof(QuantRowParams) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(p.rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kReduceThreads, 1, 1)];
    stream->endKernelCoalescing();
  });
  return out.reshape_as(q);
}

at::Tensor rms_rope_fused(
    at::Tensor x,
    at::Tensor freqs,
    at::Tensor weight,
    double eps,
    bool split_half,
    int64_t rot_dim,
    bool inplace) {
  check_mps_fp(x, "rms_rope_fused");
  check_mps_fp(weight, "rms_rope_fused:weight");
  TORCH_CHECK(x.dim() == 4 && x.size(-1) % 2 == 0);
  TORCH_CHECK(x.is_contiguous(), "rms_rope_fused requires contiguous BHND or BNHD x");
  const RopeLayout L = detect_rope_layout(x, freqs);
  const int64_t B = L.B, H = L.H, T = L.T, D = L.D;
  const int64_t rot = rot_dim > 0 ? rot_dim : D;
  TORCH_CHECK(rot % 2 == 0 && rot <= D);
  TORCH_CHECK(weight.numel() == D && weight.is_contiguous());

  auto f = prepare_freqs(freqs, B, H, T, rot / 2, L.bnhd);
  at::Tensor out = inplace ? x : at::empty_like(x);
  const int64_t rows = B * H * T;
  auto inv_rms = at::empty({rows}, x.options().dtype(at::kFloat));
  // Contiguous BHND or BNHD both reshape to (rows, D) with matching inv_rms order.
  auto x2 = x.reshape({rows, D});

  RmsInvParams ip;
  ip.rows = (uint32_t)rows;
  ip.dim = (uint32_t)D;
  ip.eps = (float)eps;
  ip.x_stride = stride_u32(x2, 0, "x");

  RmsRopeParams p;
  p.B = (uint32_t)B;
  p.H = (uint32_t)H;
  p.T = (uint32_t)T;
  p.D = (uint32_t)D;
  p.eps = (float)eps;
  fill_x_strides(x, L, p.x_stride_b, p.x_stride_h, p.x_stride_t);
  if (L.bnhd) {
    p.out_stride_b = stride_u32(out, 0, "out");
    p.out_stride_t = stride_u32(out, 1, "out");
    p.out_stride_h = stride_u32(out, 2, "out");
  } else {
    p.out_stride_b = stride_u32(out, 0, "out");
    p.out_stride_h = stride_u32(out, 1, "out");
    p.out_stride_t = stride_u32(out, 2, "out");
  }
  p.f_stride_b = f.stride_b;
  p.f_stride_h = f.stride_h;
  p.f_stride_t = f.stride_t;
  p.f_stride_pair = f.stride_pair;
  p.rot_dim = (uint32_t)rot;
  p.split_half = split_half ? 1u : 0u;
  p.bnhd = L.bnhd ? 1u : 0u;

  const std::string inv_name = std::string("rms_inv_rows_") + suffix(x.scalar_type());
  const std::string apply_name = std::string("rms_rope_apply_") + suffix(x.scalar_type());
  const std::string tail_name = std::string("rms_rope_tail_") + suffix(x.scalar_type());
  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> inv_pso = get_pipeline(device, inv_name.c_str());
  id<MTLComputePipelineState> apply_pso = get_pipeline(device, apply_name.c_str());
  id<MTLComputePipelineState> tail_pso =
      (rot < D) ? get_pipeline(device, tail_name.c_str()) : nil;
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  const NSUInteger n_pairs = (NSUInteger)(rot / 2);
  const NSUInteger tg_x = std::min(n_pairs, (NSUInteger)32);

  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [enc setComputePipelineState:inv_pso];
    set_tensor(enc, x2, 0);
    set_tensor(enc, inv_rms, 1);
    [enc setBytes:&ip length:sizeof(RmsInvParams) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kReduceThreads, 1, 1)];

    [enc setComputePipelineState:apply_pso];
    set_tensor(enc, x, 0);
    set_tensor(enc, weight, 1);
    set_tensor(enc, inv_rms, 2);
    set_tensor(enc, out, 3);
    set_tensor(enc, f.storage, 4);
    [enc setBytes:&p length:sizeof(RmsRopeParams) atIndex:5];
    [enc dispatchThreads:MTLSizeMake(n_pairs, (NSUInteger)T, (NSUInteger)(B * H))
        threadsPerThreadgroup:MTLSizeMake(tg_x, 1, 1)];

    if (tail_pso != nil) {
      [enc setComputePipelineState:tail_pso];
      set_tensor(enc, x, 0);
      set_tensor(enc, weight, 1);
      set_tensor(enc, inv_rms, 2);
      set_tensor(enc, out, 3);
      [enc setBytes:&p length:sizeof(RmsRopeParams) atIndex:4];
      const NSUInteger tail = (NSUInteger)(D - rot);
      const NSUInteger tgx = std::min(std::max(tail, (NSUInteger)1), (NSUInteger)32);
      [enc dispatchThreads:MTLSizeMake(tail, (NSUInteger)T, (NSUInteger)(B * H))
          threadsPerThreadgroup:MTLSizeMake(tgx, 1, 1)];
    }
    stream->endKernelCoalescing();
  });
  return out;
}

static uint32_t fp8_fmt_code(c10::ScalarType dt) {
  if (dt == at::kFloat8_e4m3fn) {
    return 0u;
  }
  if (dt == at::kFloat8_e5m2) {
    return 1u;
  }
  TORCH_CHECK(false, "unsupported fp8 dtype");
}

at::Tensor quantize_per_tensor_fp8(at::Tensor x, at::Tensor scale, int64_t output_dtype_code) {
  TORCH_CHECK(x.device().is_mps());
  TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kBFloat16 || x.dtype() == at::kFloat);
  TORCH_CHECK(scale.device().is_mps() && scale.numel() == 1);
  x = x.contiguous();
  scale = scale.to(at::kFloat).contiguous();

  c10::ScalarType out_dt = output_dtype_code == 6 ? at::kFloat8_e5m2 : at::kFloat8_e4m3fn;
  // Allocate as the FP8 dtype when possible so the returned tensor carries the
  // right semantic type; fill through a uint8 view (MPS e5m2 storage path).
  auto out = at::empty(x.sizes(), x.options().dtype(out_dt));
  auto out_u8 = out.view(at::kByte);

  Fp8Params p;
  p.numel = (uint32_t)x.numel();
  p.fmt = fp8_fmt_code(out_dt);

  const char* kname = x.dtype() == at::kFloat
      ? "quantize_fp8_fp32"
      : (x.dtype() == at::kBFloat16 ? "quantize_fp8_bf16" : "quantize_fp8_fp16");
  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = get_pipeline(device, kname);
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  const NSUInteger nthreads = ((p.numel + kFp8Threads - 1) / kFp8Threads) * kFp8Threads;

  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [enc setComputePipelineState:pipeline];
    set_tensor(enc, x, 0);
    set_tensor(enc, scale, 1);
    set_tensor(enc, out_u8, 2);
    [enc setBytes:&p length:sizeof(Fp8Params) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(std::max<NSUInteger>(nthreads, kFp8Threads), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kFp8Threads, 1, 1)];
    stream->endKernelCoalescing();
  });
  return out;
}

at::Tensor dequantize_per_tensor_fp8(at::Tensor x, at::Tensor scale, int64_t output_dtype_code) {
  TORCH_CHECK(x.device().is_mps());
  TORCH_CHECK(scale.device().is_mps() && scale.numel() == 1);
  TORCH_CHECK(x.dtype() == at::kFloat8_e4m3fn || x.dtype() == at::kFloat8_e5m2);
  scale = scale.to(at::kFloat).reshape({}).contiguous();

  c10::ScalarType out_dt =
      output_dtype_code == 0 ? at::kFloat : (output_dtype_code == 2 ? at::kBFloat16 : at::kHalf);
  auto x_u8 = x.view(at::kByte).contiguous();
  auto out = at::empty(x.sizes(), x.options().dtype(out_dt));

  Fp8Params p;
  p.numel = (uint32_t)x.numel();
  p.fmt = fp8_fmt_code(x.scalar_type());

  const char* kname = out_dt == at::kFloat
      ? "dequantize_fp8_to_fp32"
      : (out_dt == at::kBFloat16 ? "dequantize_fp8_to_bf16" : "dequantize_fp8_to_fp16");
  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = get_pipeline(device, kname);
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  const NSUInteger nthreads = ((p.numel + kFp8Threads - 1) / kFp8Threads) * kFp8Threads;

  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [enc setComputePipelineState:pipeline];
    set_tensor(enc, x_u8, 0);
    set_tensor(enc, scale, 1);
    set_tensor(enc, out, 2);
    [enc setBytes:&p length:sizeof(Fp8Params) atIndex:3];
    [enc dispatchThreads:MTLSizeMake(std::max<NSUInteger>(nthreads, kFp8Threads), 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kFp8Threads, 1, 1)];
    stream->endKernelCoalescing();
  });
  return out;
}

at::Tensor svdquant_dequant_w4a4(
    at::Tensor packed,
    at::Tensor scales,
    bool act_unsigned) {
  TORCH_CHECK(packed.device().is_mps() && packed.dtype() == at::kChar);
  check_mps_fp(scales, "svdquant_dequant:scales");
  TORCH_CHECK(packed.dim() == 2 && scales.dim() == 2);
  packed = packed.contiguous();
  scales = scales.contiguous();
  const int64_t rows = packed.size(0);
  const int64_t k_half = packed.size(1);
  const int64_t k = k_half * 2;
  TORCH_CHECK(k % kSvdGroupSize == 0);
  const int64_t groups = k / kSvdGroupSize;
  TORCH_CHECK(scales.size(0) == groups && scales.size(1) == rows);

  auto out = at::empty({rows, k}, scales.options());
  SvdDequantParams p;
  p.rows = (uint32_t)rows;
  p.k = (uint32_t)k;
  p.groups = (uint32_t)groups;
  p.packed_stride = stride_u32(packed, 0, "packed");
  p.scale_stride = stride_u32(scales, 0, "scales");
  p.out_stride = stride_u32(out, 0, "out");
  p.unsigned_act = act_unsigned ? 1u : 0u;

  std::string kname = std::string("svdquant_dequant_") + suffix(scales.scalar_type());
  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = get_pipeline(device, kname.c_str());
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [enc setComputePipelineState:pipeline];
    set_tensor(enc, packed, 0);
    set_tensor(enc, scales, 1);
    set_tensor(enc, out, 2);
    [enc setBytes:&p length:sizeof(SvdDequantParams) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kReduceThreads, 1, 1)];
    stream->endKernelCoalescing();
  });
  return out;
}

std::tuple<at::Tensor, at::Tensor> svdquant_quantize_w4a4(
    at::Tensor x,
    at::Tensor smooth,
    bool act_unsigned) {
  check_mps_fp(x, "svdquant_quantize:x");
  check_mps_fp(smooth, "svdquant_quantize:smooth");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous());
  TORCH_CHECK(smooth.numel() == x.size(1) && smooth.is_contiguous());
  const int64_t rows = x.size(0);
  const int64_t k = x.size(1);
  TORCH_CHECK(k % kSvdGroupSize == 0);
  const int64_t groups = k / kSvdGroupSize;

  auto packed = at::empty({rows, k / 2}, x.options().dtype(at::kChar));
  auto scales = at::empty({groups, rows}, x.options());

  SvdQuantParams p;
  p.rows = (uint32_t)rows;
  p.k = (uint32_t)k;
  p.groups = (uint32_t)groups;
  p.x_stride = stride_u32(x, 0, "x");
  p.packed_stride = stride_u32(packed, 0, "packed");
  p.scale_stride = stride_u32(scales, 0, "scales");
  p.unsigned_act = act_unsigned ? 1u : 0u;
  p.qmax = act_unsigned ? 15u : 7u;

  std::string kname = std::string("svdquant_quantize_") + suffix(x.scalar_type());
  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = get_pipeline(device, kname.c_str());
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [enc setComputePipelineState:pipeline];
    set_tensor(enc, x, 0);
    set_tensor(enc, smooth, 1);
    set_tensor(enc, packed, 2);
    set_tensor(enc, scales, 3);
    [enc setBytes:&p length:sizeof(SvdQuantParams) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kReduceThreads, 1, 1)];
    stream->endKernelCoalescing();
  });
  return {packed, scales};
}

static void dispatch_flat(
    id<MTLComputePipelineState> pipeline,
    at::mps::MPSStream* stream,
    uint32_t numel,
    void (^bind)(id<MTLComputeCommandEncoder>)) {
  if (numel == 0) {
    return;
  }
  const NSUInteger tg = 256;
  const NSUInteger nthreads = std::min((NSUInteger)numel, tg * 256);
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [enc setComputePipelineState:pipeline];
    bind(enc);
    [enc dispatchThreads:MTLSizeMake(nthreads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(std::min(nthreads, tg), 1, 1)];
    stream->endKernelCoalescing();
  });
}

at::Tensor quantize_int8_tensorwise(
    at::Tensor x, at::Tensor scale, bool stochastic = false, int64_t seed = 0) {
  check_mps_fp(x, "quantize_int8_tensorwise");
  TORCH_CHECK(scale.device().is_mps() && scale.numel() == 1);
  x = x.contiguous();
  scale = scale.to(at::kFloat).contiguous();
  auto q = at::empty(x.sizes(), x.options().dtype(at::kChar));

  TensorQuantParams p{};
  p.numel = (uint32_t)x.numel();
  set_tensor_quant_seed(p, stochastic, seed);

  std::string kname = std::string("quantize_int8_tensorwise_") + suffix(x.scalar_type());
  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = get_pipeline(device, kname.c_str());
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  dispatch_flat(pipeline, stream, p.numel, ^(id<MTLComputeCommandEncoder> enc) {
    set_tensor(enc, x, 0);
    set_tensor(enc, scale, 1);
    set_tensor(enc, q, 2);
    [enc setBytes:&p length:sizeof(TensorQuantParams) atIndex:3];
  });
  return q;
}

at::Tensor dequantize_int8_simple_dtype(
    at::Tensor q, at::Tensor scale, int64_t output_dtype_code) {
  TORCH_CHECK(q.device().is_mps() && q.dtype() == at::kChar);
  TORCH_CHECK(scale.device().is_mps());
  q = q.contiguous();
  scale = scale.to(at::kFloat).contiguous();

  c10::ScalarType out_dt = at::kFloat;
  const char* kname = "dequantize_int8_flat_fp32";
  if (output_dtype_code == 1) {
    out_dt = at::kHalf;
    kname = "dequantize_int8_flat_fp16";
  } else if (output_dtype_code == 2) {
    out_dt = at::kBFloat16;
    kname = "dequantize_int8_flat_bf16";
  } else {
    TORCH_CHECK(output_dtype_code == 0, "unsupported dequant output dtype");
  }

  auto out = at::empty(q.sizes(), q.options().dtype(out_dt));
  DequantFlatParams p;
  p.numel = (uint32_t)q.numel();
  p.inner_dim = q.dim() > 0 ? (uint32_t)q.size(-1) : 1u;
  if (scale.numel() == 1) {
    p.scale_mode = 0u;
  } else if (scale.sizes() == q.sizes()) {
    p.scale_mode = 1u;
  } else if (q.dim() > 0 && scale.dim() == q.dim() && scale.size(-1) == 1) {
    bool ok = true;
    for (int64_t d = 0; d < q.dim() - 1; ++d) {
      if (scale.size(d) != q.size(d)) {
        ok = false;
        break;
      }
    }
    TORCH_CHECK(ok, "dequantize_int8_simple_dtype: unsupported scale shape");
    p.scale_mode = 2u;
  } else {
    TORCH_CHECK(false, "dequantize_int8_simple_dtype: unsupported scale shape");
  }

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = get_pipeline(device, kname);
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  dispatch_flat(pipeline, stream, p.numel, ^(id<MTLComputeCommandEncoder> enc) {
    set_tensor(enc, q, 0);
    set_tensor(enc, scale, 1);
    set_tensor(enc, out, 2);
    [enc setBytes:&p length:sizeof(DequantFlatParams) atIndex:3];
  });
  return out;
}

at::Tensor stochastic_round_fp8(at::Tensor x, at::Tensor rng, int64_t output_dtype_code) {
  TORCH_CHECK(x.device().is_mps());
  TORCH_CHECK(x.dtype() == at::kHalf || x.dtype() == at::kBFloat16 || x.dtype() == at::kFloat);
  TORCH_CHECK(rng.device().is_mps() && rng.dtype() == at::kByte);
  TORCH_CHECK(rng.is_contiguous(), "stochastic_round_fp8 requires contiguous rng");
  TORCH_CHECK(rng.sizes() == x.sizes());
  x = x.contiguous();

  c10::ScalarType out_dt = output_dtype_code == 6 ? at::kFloat8_e5m2 : at::kFloat8_e4m3fn;
  Fp8Params p;
  p.numel = (uint32_t)x.numel();
  p.fmt = fp8_fmt_code(out_dt);

  const char* kname = x.dtype() == at::kFloat
      ? "stochastic_round_fp8_fp32"
      : (x.dtype() == at::kBFloat16 ? "stochastic_round_fp8_bf16" : "stochastic_round_fp8_fp16");

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = get_pipeline(device, kname);
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  dispatch_flat(pipeline, stream, p.numel, ^(id<MTLComputeCommandEncoder> enc) {
    set_tensor(enc, x, 0);
    set_tensor(enc, rng, 1);
    [enc setBytes:&p length:sizeof(Fp8Params) atIndex:2];
  });
  return rng.view(out_dt);
}

struct MxParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
};

struct Nvfp4MmParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t a_packed_stride;
  uint32_t b_packed_stride;
  uint32_t a_scale_stride;
  uint32_t b_scale_stride;
  uint32_t out_stride;
  float tensor_scale_a;
  float tensor_scale_b;
  uint32_t hi_first;
};

struct SvdMmParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t groups;
  uint32_t a_packed_stride;
  uint32_t w_packed_stride;
  uint32_t a_scale_stride;
  uint32_t w_scale_stride;
  uint32_t out_stride;
  uint32_t act_unsigned;
};

struct AwqParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t group_size;
  uint32_t x_stride;
  uint32_t q_stride;
  uint32_t scale_stride;
  uint32_t out_stride;
  uint32_t has_bias;
};

// Tiled fused GEMMs use 8x8 threadgroups (must match fused_quant_mm.metal).
constexpr NSUInteger kGemmTileM = 8;
constexpr NSUInteger kGemmTileN = 8;

static void dispatch_tiled_gemm(
    id<MTLComputePipelineState> pipeline,
    at::mps::MPSStream* stream,
    NSUInteger N,
    NSUInteger M,
    void (^bind)(id<MTLComputeCommandEncoder>)) {
  if (N == 0 || M == 0) {
    return;
  }
  const NSUInteger groups_n = (N + kGemmTileN - 1) / kGemmTileN;
  const NSUInteger groups_m = (M + kGemmTileM - 1) / kGemmTileM;
  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [enc setComputePipelineState:pipeline];
    bind(enc);
    [enc dispatchThreadgroups:MTLSizeMake(groups_n, groups_m, 1)
        threadsPerThreadgroup:MTLSizeMake(kGemmTileN, kGemmTileM, 1)];
    stream->endKernelCoalescing();
  });
}

at::Tensor scaled_mm_nvfp4_fused(
    at::Tensor a_packed,
    at::Tensor b_packed,
    at::Tensor a_scales_u8,
    at::Tensor b_scales_u8,
    double tensor_scale_a,
    double tensor_scale_b,
    bool hi_first) {
  TORCH_CHECK(a_packed.device().is_mps() && b_packed.device().is_mps());
  TORCH_CHECK(a_packed.dim() == 2 && b_packed.dim() == 2);
  TORCH_CHECK(a_packed.size(1) == b_packed.size(1));
  const int64_t M = a_packed.size(0);
  const int64_t N = b_packed.size(0);
  const int64_t K = a_packed.size(1) * 2;
  TORCH_CHECK(K % 16 == 0);
  TORCH_CHECK(a_scales_u8.sizes() == at::IntArrayRef({M, K / 16}));
  TORCH_CHECK(b_scales_u8.sizes() == at::IntArrayRef({N, K / 16}));

  a_packed = a_packed.contiguous();
  b_packed = b_packed.contiguous();
  a_scales_u8 = a_scales_u8.contiguous();
  b_scales_u8 = b_scales_u8.contiguous();
  auto out = at::empty({M, N}, a_packed.options().dtype(at::kHalf));

  Nvfp4MmParams p;
  p.M = (uint32_t)M;
  p.N = (uint32_t)N;
  p.K = (uint32_t)K;
  p.a_packed_stride = (uint32_t)a_packed.stride(0);
  p.b_packed_stride = (uint32_t)b_packed.stride(0);
  p.a_scale_stride = (uint32_t)a_scales_u8.stride(0);
  p.b_scale_stride = (uint32_t)b_scales_u8.stride(0);
  p.out_stride = (uint32_t)out.stride(0);
  p.tensor_scale_a = (float)tensor_scale_a;
  p.tensor_scale_b = (float)tensor_scale_b;
  p.hi_first = hi_first ? 1u : 0u;

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = get_pipeline(device, "scaled_mm_nvfp4_fused");
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  dispatch_tiled_gemm(pipeline, stream, (NSUInteger)N, (NSUInteger)M, ^(id<MTLComputeCommandEncoder> enc) {
    set_tensor(enc, a_packed, 0);
    set_tensor(enc, b_packed, 1);
    set_tensor(enc, a_scales_u8, 2);
    set_tensor(enc, b_scales_u8, 3);
    set_tensor(enc, out, 4);
    [enc setBytes:&p length:sizeof(Nvfp4MmParams) atIndex:5];
  });
  return out;
}

at::Tensor scaled_mm_svdquant_fused(
    at::Tensor a_packed,
    at::Tensor w_packed,
    at::Tensor a_scales,
    at::Tensor w_scales,
    bool act_unsigned) {
  TORCH_CHECK(a_packed.device().is_mps() && w_packed.device().is_mps());
  TORCH_CHECK(a_scales.dtype() == at::kHalf && w_scales.dtype() == at::kHalf);
  const int64_t M = a_packed.size(0);
  const int64_t N = w_packed.size(0);
  const int64_t K = a_packed.size(1) * 2;
  TORCH_CHECK(K % 64 == 0);
  const int64_t groups = K / 64;
  TORCH_CHECK(a_scales.sizes() == at::IntArrayRef({groups, M}));
  TORCH_CHECK(w_scales.sizes() == at::IntArrayRef({groups, N}));

  a_packed = a_packed.contiguous();
  w_packed = w_packed.contiguous();
  a_scales = a_scales.contiguous();
  w_scales = w_scales.contiguous();
  auto out = at::empty({M, N}, a_scales.options().dtype(at::kHalf));

  SvdMmParams p;
  p.M = (uint32_t)M;
  p.N = (uint32_t)N;
  p.K = (uint32_t)K;
  p.groups = (uint32_t)groups;
  p.a_packed_stride = (uint32_t)a_packed.stride(0);
  p.w_packed_stride = (uint32_t)w_packed.stride(0);
  p.a_scale_stride = (uint32_t)a_scales.stride(0);
  p.w_scale_stride = (uint32_t)w_scales.stride(0);
  p.out_stride = (uint32_t)out.stride(0);
  p.act_unsigned = act_unsigned ? 1u : 0u;

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = get_pipeline(device, "scaled_mm_svdquant_fused");
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  dispatch_tiled_gemm(pipeline, stream, (NSUInteger)N, (NSUInteger)M, ^(id<MTLComputeCommandEncoder> enc) {
    set_tensor(enc, a_packed, 0);
    set_tensor(enc, w_packed, 1);
    set_tensor(enc, a_scales, 2);
    set_tensor(enc, w_scales, 3);
    set_tensor(enc, out, 4);
    [enc setBytes:&p length:sizeof(SvdMmParams) atIndex:5];
  });
  return out;
}

at::Tensor gemv_awq_w4a16_fused(
    at::Tensor x,
    at::Tensor qweight,
    at::Tensor wscales,
    at::Tensor wzeros,
    c10::optional<at::Tensor> bias,
    int64_t group_size) {
  TORCH_CHECK(x.device().is_mps() && qweight.device().is_mps());
  TORCH_CHECK(x.dtype() == at::kHalf && wscales.dtype() == at::kHalf);
  x = x.contiguous();
  auto x2d = x.dim() == 1 ? x.unsqueeze(0) : x.reshape({-1, x.size(-1)}).contiguous();
  const int64_t M = x2d.size(0);
  const int64_t K = x2d.size(1);
  const int64_t N = qweight.size(0);
  TORCH_CHECK(K % group_size == 0);
  TORCH_CHECK(qweight.size(1) * 2 == K);
  TORCH_CHECK(wscales.sizes() == at::IntArrayRef({K / group_size, N}));
  TORCH_CHECK(wzeros.sizes() == wscales.sizes());

  qweight = qweight.contiguous();
  wscales = wscales.contiguous();
  wzeros = wzeros.contiguous();
  auto out = at::empty({M, N}, x.options().dtype(at::kHalf));
  at::Tensor bias_t;
  const bool has_bias = bias.has_value() && bias->defined();
  if (has_bias) {
    bias_t = bias->to(x.device(), at::kHalf).contiguous();
  } else {
    bias_t = at::empty({1}, x.options().dtype(at::kHalf));
  }

  AwqParams p;
  p.M = (uint32_t)M;
  p.N = (uint32_t)N;
  p.K = (uint32_t)K;
  p.group_size = (uint32_t)group_size;
  p.x_stride = (uint32_t)x2d.stride(0);
  p.q_stride = (uint32_t)qweight.stride(0);
  p.scale_stride = (uint32_t)wscales.stride(0);
  p.out_stride = (uint32_t)out.stride(0);
  p.has_bias = has_bias ? 1u : 0u;

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = get_pipeline(device, "gemv_awq_w4a16_fused");
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  dispatch_tiled_gemm(pipeline, stream, (NSUInteger)N, (NSUInteger)M, ^(id<MTLComputeCommandEncoder> enc) {
    set_tensor(enc, x2d, 0);
    set_tensor(enc, qweight, 1);
    set_tensor(enc, wscales, 2);
    set_tensor(enc, wzeros, 3);
    set_tensor(enc, bias_t, 4);
    set_tensor(enc, out, 5);
    [enc setBytes:&p length:sizeof(AwqParams) atIndex:6];
  });

  if (x.dim() == 1) {
    return out.reshape({N});
  }
  auto leading = x.sizes().vec();
  leading.back() = N;
  return out.reshape(leading);
}

at::Tensor scaled_mm_mxfp8_tensorops(
    at::Tensor a,
    at::Tensor b,
    at::Tensor a_scales,
    at::Tensor b_scales) {
  // A (M,K) FP8, B (N,K) FP8, unswizzled E8M0 scales (M,K/32) and (N,K/32).
  // Native TensorOps matmul2d with MX scale planes; C = A @ B.T in fp16.
  TORCH_CHECK(a.device().is_mps() && b.device().is_mps(), "mxfp8 tensorops: mps required");
  TORCH_CHECK(a_scales.device().is_mps() && b_scales.device().is_mps(), "mxfp8 tensorops: scales on mps");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "mxfp8 tensorops: 2D matrices");
  TORCH_CHECK(a.size(1) == b.size(1), "mxfp8 tensorops: K mismatch");
  TORCH_CHECK(a.size(1) % 32 == 0, "mxfp8 tensorops: K must be multiple of 32");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "mxfp8 tensorops: contiguous data");
  TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous(), "mxfp8 tensorops: contiguous scales");

  const int64_t M = a.size(0);
  const int64_t N = b.size(0);
  const int64_t K = a.size(1);
  TORCH_CHECK(a_scales.sizes() == at::IntArrayRef({M, K / 32}), "mxfp8 tensorops: bad A scales shape");
  TORCH_CHECK(b_scales.sizes() == at::IntArrayRef({N, K / 32}), "mxfp8 tensorops: bad B scales shape");

  auto out = at::zeros({M, N}, a.options().dtype(at::kHalf));

  MxParams p;
  p.M = (uint32_t)M;
  p.N = (uint32_t)N;
  p.K = (uint32_t)K;

  id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
  id<MTLComputePipelineState> pipeline = get_pipeline(device, "scaled_mm_mxfp8_tensorops");
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

  const NSUInteger tgM = (NSUInteger)((M + 31) / 32);
  const NSUInteger tgN = (NSUInteger)((N + 31) / 32);
  // 4 simdgroups × 32 threads
  const NSUInteger threads_per_tg = 128;

  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)stream->commandEncoder();
    [enc setComputePipelineState:pipeline];
    set_tensor(enc, a, 0);
    set_tensor(enc, a_scales, 1);
    set_tensor(enc, b, 2);
    set_tensor(enc, b_scales, 3);
    set_tensor(enc, out, 4);
    [enc setBytes:&p length:sizeof(MxParams) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(tgN, tgM, 1)
        threadsPerThreadgroup:MTLSizeMake(threads_per_tg, 1, 1)];
    stream->endKernelCoalescing();
  });
  return out;
}

bool metal_available() {
  return at::mps::MPSDevice::getInstance()->device() != nullptr;
}

bool mxfp8_tensorops_available() {
  try {
    id<MTLDevice> device = (id<MTLDevice>)at::mps::MPSDevice::getInstance()->device();
    if (device == nil) {
      return false;
    }
    return get_pipeline(device, "scaled_mm_mxfp8_tensorops") != nil;
  } catch (...) {
    return false;
  }
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("residual_rms_norm", &residual_rms_norm);
  m.def("rms_gated_residual", &rms_gated_residual);
  m.def("adaln", &adaln);
  m.def("apply_rope1_", &apply_rope1_);
  m.def("apply_rope_split_", &apply_rope_split_);
  m.def("rms_rope_fused", &rms_rope_fused);
  m.def(
      "quantize_int8_rowwise",
      &quantize_int8_rowwise,
      py::arg("x"),
      py::arg("stochastic") = false,
      py::arg("seed") = 0);
  m.def(
      "quantize_int4_rowwise",
      &quantize_int4_rowwise,
      py::arg("x"),
      py::arg("stochastic") = false,
      py::arg("seed") = 0);
  m.def("dequantize_int4_rowwise", &dequantize_int4_rowwise);
  m.def("dequantize_int8_simple", &dequantize_int8_simple);
  m.def(
      "quantize_int8_tensorwise",
      &quantize_int8_tensorwise,
      py::arg("x"),
      py::arg("scale"),
      py::arg("stochastic") = false,
      py::arg("seed") = 0);
  m.def("dequantize_int8_simple_dtype", &dequantize_int8_simple_dtype);
  m.def("quantize_per_tensor_fp8", &quantize_per_tensor_fp8);
  m.def("dequantize_per_tensor_fp8", &dequantize_per_tensor_fp8);
  m.def("stochastic_round_fp8", &stochastic_round_fp8);
  m.def("svdquant_dequant_w4a4", &svdquant_dequant_w4a4);
  m.def("svdquant_quantize_w4a4", &svdquant_quantize_w4a4);
  m.def("scaled_mm_mxfp8_tensorops", &scaled_mm_mxfp8_tensorops);
  m.def("scaled_mm_nvfp4_fused", &scaled_mm_nvfp4_fused);
  m.def("scaled_mm_svdquant_fused", &scaled_mm_svdquant_fused);
  m.def("gemv_awq_w4a16_fused", &gemv_awq_w4a16_fused);
  m.def("metal_available", &metal_available);
  m.def("mxfp8_tensorops_available", &mxfp8_tensorops_available);
}
