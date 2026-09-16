// SPDX-License-Identifier: Apache-2.0
// Native scheduling of the validated Draft SM89 H3 kernels.
#include "draft_dispatch.h"
#include "include/api.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace draft_native {
int64_t sm89_route_workspace_bytes(int64_t, int64_t, int64_t, int, cudaStream_t);
}

namespace comfy::draft {
namespace {
using draft_native::check;
using Shape = draft_native::IntArrayRef;
constexpr auto H = DType::Half, F = DType::Float, I = DType::Int;
constexpr auto C = DType::Char, U = DType::Byte, E = DType::Float8_e4m3fn;

// Preserved kernel host wrappers consume these preplanned workspace views.
class KernelContext {
  draft_native::ExecutionContext context_;
  draft_native::ExecutionContext *previous_;
public:
  KernelContext(int device, cudaStream_t stream,
                std::initializer_list<Tensor> buffers)
      : previous_(draft_native::active_context) {
    context_.device = device;
    context_.stream = stream;
    context_.buffers = buffers;
    draft_native::active_context = &context_;
  }
  ~KernelContext() { draft_native::active_context = previous_; }
  void finish() const {
    check(context_.next == context_.buffers.size(),
          "SM89 kernel allocation recipe changed");
  }
};

struct Plan {
  Workspace workspace;
  int prefix_blocks;
  bool prefix_int8, prefix_query_int8, active_int8, active_fp16;
  int64_t n16, n8;
  Region qp, kp, qm, km, q16, k16, v16, q8, k8, v8, qs, ks, vs;
  Region mean, key_sums, value_amax, valid_k, qsecond, ksecond;
  Region probability, max_logits, draft_scratch;
  Region precision, logical_ids, sorted_ids, input_keys, sorted_keys;
  Region low, middle, high, route_scratch;
  Region ids, physical_low, physical_middle, physical_high;
  Region prefix_q, prefix_scale, prefix_out, video_out, lse;

  explicit Plan(const Config &c) {
    validate(c);
    check(ada_serves(c.architecture) && c.draft_proxy == 0 &&
              c.ratios[0] == 0.0 && c.ratios[2] == 0.0,
          "the SM89 kernel set supports mean/Jensen/maxpool and INT8/FP16 only");
    check(c.prefix_kv == -1 || c.prefix_kv == 1 || c.prefix_kv == 3,
          "SM89-set prefix KV precision must be auto, INT8, or FP16");
    check(c.prefix_query == -1 || c.prefix_query == 1 || c.prefix_query == 3,
          "SM89-set prefix query precision must be auto, INT8, or FP16");
    prefix_blocks = (c.prefix_tokens + 63) / 64;
    prefix_int8 = !c.prefix_tokens || resolved_prefix_kv(c) == 1;
    prefix_query_int8 = c.prefix_tokens && resolved_prefix_query(c) == 1;
    active_int8 = c.ratios[1] > 0 || (c.prefix_tokens && prefix_int8);
    active_fp16 = c.ratios[3] > 0 || !prefix_int8;

    const auto counts = route_budgets(c);
    n16 = counts[0]; n8 = counts[1];
    check(counts[2] == 0, "SM89 route unexpectedly has an NVFP4 budget");
    check(!c.anchors || c.anchor_count <= (n8 ? n8 : n16),
          "anchor_count exceeds lowest-precision budget");

    const int64_t b = c.batch, h = c.heads, d = c.dim, r = c.video_blocks;
    const int64_t video = r * c.query_block;
    const int64_t key = video + int64_t(prefix_blocks) * 64;
    const int64_t stages = key / 64;
    auto add = [&](Shape s, DType t) { return workspace.add(s, t, c.device); };
    auto packed = [&](int64_t n) { return Shape{b,h,n,d}; };
    const Shape pools = packed(r), rows{b,h,r}, map{b,h,r,r}, empty{0};
    qp = add(pools,H); kp = add(pools,H);
    qm = add(c.maxpool_weight != 0 ? pools : empty,H);
    km = add(c.maxpool_weight != 0 ? pools : empty,H);
    q16 = add(packed(video),H); k16 = add(packed(key),H); v16 = add(packed(key),H);
    q8 = add(packed(video),C); k8 = add(packed(key),C);
    v8 = add({b,h,d,((key+127)/128)*128},E);
    qs = add(rows,F); ks = add({b,h,stages},F); vs = add({b,h,d},F);
    mean = add(c.smooth_k ? Shape{b,h,d} : empty,H);
    key_sums = add(c.smooth_k ? packed(stages) : empty,F);
    value_amax = add(packed(stages),F);
    valid_k = add({b,stages},I);
    qsecond = add(c.jensen ? pools : empty,H);
    ksecond = add(c.jensen ? pools : empty,H);
    probability = add(map,H);
    max_logits = add(c.maxpool_weight > 0 && c.maxpool_weight < 1 ? map : empty,H);
    const size_t draft_bytes = c.maxpool_weight == 0
        ? draft_workspace_bytes(c.batch,c.heads,c.video_blocks,c.dim,c.jensen) : 0;
    check(draft_bytes <= INT64_MAX, "SM89 draft scratch overflow");
    draft_scratch = add({int64_t(draft_bytes)},U);
    precision = add(map,U); logical_ids = add(map,I); sorted_ids = add(map,I);
    input_keys = add(map,I); sorted_keys = add(map,I);
    low = add(rows,I); middle = add(rows,I); high = add(rows,I);
    route_scratch = add({draft_native::sm89_route_workspace_bytes(
        b,h,r,c.device,nullptr)},U);
    ids = add({b,h,r,stages},I);
    physical_low = add(rows,I); physical_middle = add(rows,I); physical_high = add(rows,I);
    prefix_q = add(prefix_query_int8 ? packed(prefix_blocks*64) : empty,C);
    prefix_scale = add(prefix_query_int8 ? Shape{b,h,prefix_blocks} : empty,F);
    prefix_out = add(prefix_query_int8 ? packed(prefix_blocks*64) : empty,H);
    video_out = add(packed(video),H); lse = add(empty,F);
  }
};
} // namespace

size_t workspace_sm89(const Config &c) { return Plan(c).workspace.bytes; }

void execute_sm89(const Config &c, const Inputs &in, void *workspace,
                  size_t bytes, cudaStream_t stream) {
  const Plan p(c);
  check(bytes >= p.workspace.bytes && (workspace || !p.workspace.bytes),
        "insufficient SM89 workspace");
  draft_native::cuda::CUDAGuard device_guard(c.device);
  auto t = [&](const Region &r) { return view(r, workspace); };
  const auto q = t(p.q16), k = t(p.k16), v = t(p.v16);
  const auto qi = t(p.q8), ki = t(p.k8), vi = t(p.v8);
  const auto qs = t(p.qs), ks = t(p.ks), vs = t(p.vs), mean = t(p.mean);
  const auto valid = t(p.valid_k), ids = t(p.ids);
  const auto mid = t(p.physical_middle), hi = t(p.physical_high);
  const double scale = 1.0 / std::sqrt(double(c.dim));
  {
    KernelContext ctx(c.device,stream,{t(p.qp),t(p.kp),t(p.qm),t(p.km),q,k,v,
        qi,ki,vi,qs,ks,vs,mean,t(p.key_sums),t(p.value_amax)});
    prepare_h3_sm89_int8_operands(in.q,in.k,in.v,in.indices,in.slot_valid,
        in.counts,c.prefix_tokens,c.query_block,c.smooth_k,c.maxpool_weight != 0);
    ctx.finish();
  }
  valid_key_counts(c,in.counts,valid,stream);
  Tensor prefix = in.prefix_output;
  if (p.prefix_query_int8) {
    {
      KernelContext ctx(c.device,stream,{t(p.prefix_q),t(p.prefix_scale)});
      prepare_sm89_prefix_q_int8(in.q,c.prefix_tokens);
      ctx.finish();
    }
    KernelContext ctx(c.device,stream,{t(p.prefix_out)});
    prefix = sm89_q64_prefix_int8_attention_forward(t(p.prefix_q),ki,vi,
        t(p.prefix_scale),ks,vs,valid,c.prefix_tokens,scale);
    ctx.finish();
  }
  if (c.jensen) {
    moments(q,in.counts,c.query_block,t(p.qsecond),stream);
    moments(k.narrow(2,p.prefix_blocks*64,q.size(2)),in.counts,
            c.query_block,t(p.ksecond),stream);
  }
  if (c.maxpool_weight == 0) {
    draft_probability(t(p.qp),t(p.kp),t(p.qsecond),t(p.ksecond),t(p.probability),
        t(p.draft_scratch).pointer,size_t(p.draft_scratch.metadata.numel()),stream);
  } else if (c.maxpool_weight == 1) {
    KernelContext ctx(c.device,stream,{t(p.probability)});
    sm89_h3_draft_probability(t(p.qp),t(p.kp),t(p.qm),t(p.km),c.maxpool_weight);
    ctx.finish();
  } else {
    KernelContext ctx(c.device,stream,{t(p.probability),t(p.max_logits)});
    sm89_h3_draft_probability(t(p.qp),t(p.kp),t(p.qm),t(p.km),c.maxpool_weight);
    ctx.finish();
  }
  {
    KernelContext ctx(c.device,stream,{t(p.precision),t(p.logical_ids),t(p.sorted_ids),
        t(p.input_keys),t(p.sorted_keys),t(p.low),t(p.middle),t(p.high),t(p.route_scratch)});
    sm89_h3_route_precision(t(p.probability),p.n16,p.n8,0,
        c.anchors ? std::optional<Tensor>(in.anchors) : std::nullopt,
        c.anchors ? std::optional<Tensor>(in.anchor_ids) : std::nullopt,c.anchor_count);
    ctx.finish();
  }
  {
    KernelContext ctx(c.device,stream,{ids,t(p.physical_low),mid,hi});
    sm89_h3_materialize_route(t(p.logical_ids),t(p.low),t(p.middle),t(p.high),
        c.query_block,p.prefix_blocks,p.prefix_int8 ? 1 : 2,p.prefix_int8,true);
    ctx.finish();
  }
  {
    KernelContext ctx(c.device,stream,{t(p.video_out),t(p.lse)});
    const bool q64 = c.query_block == 64;
    const int fp16_prefix = p.prefix_int8 ? 0 : p.prefix_blocks;
    if (!p.active_int8) {
      (q64 ? k64_fp16_attention_forward : q128_k64_fp16_attention_forward)(
          q,k,v,ids,hi,valid,scale);
    } else if (!p.active_fp16 && c.smooth_k) {
      (q64 ? k64_smooth_fp8_attention_forward : q128_k64_smooth_fp8_attention_forward)(
          qi,ki,vi,q,mean,ids,mid,qs,ks,vs,valid,scale);
    } else if (!p.active_fp16) {
      (q64 ? k64_fp8_attention_forward : q128_k64_fp8_attention_forward)(
          qi,ki,vi,ids,mid,qs,ks,vs,valid,scale);
    } else if (c.smooth_k) {
      (q64 ? k64_smooth_mixed_attention_forward : q128_k64_smooth_mixed_attention_forward)(
          qi,ki,vi,q,k,v,mean,ids,mid,ids,hi,qs,ks,vs,valid,fp16_prefix,scale);
    } else {
      (q64 ? k64_mixed_attention_forward : q128_k64_mixed_attention_forward)(
          qi,ki,vi,q,k,v,ids,mid,ids,hi,qs,ks,vs,valid,fp16_prefix,scale);
    }
    ctx.finish();
  }
  // The assembly kernel writes caller-owned BTHD directly, including empty-row
  // positive-zero handling and the original-dtype dense prefix overwrite.
  {
    KernelContext ctx(c.device,stream,{in.output});
    assemble_h3_k64_output(prefix,t(p.video_out),in.inverse,c.input_type,
                           mid,hi,std::nullopt,c.query_block);
    ctx.finish();
  }
}
} // namespace comfy::draft
