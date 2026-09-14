// SPDX-License-Identifier: Apache-2.0
// Native orchestration of the pinned Anemoi 4e85afba SM120 kernels.
#include "anemoi_dispatch.h"
#include "include/attention_api.h"
#include "include/preparation.h"
#include "include/route_draft.h"
#include <algorithm>
#include <cmath>

namespace comfy::anemoi {
namespace {
namespace prep = ::anemoi::sm120::preparation;
using anemoi_native::check;
using anemoi_native::cuda_check;

prep::TensorView preparation_view(const Tensor &t) {
  prep::TensorView v;
  v.data = t.pointer;
  v.ndim = int(t.dim());
  v.device = t.device();
  check(v.ndim <= 4, "SM120 preparation descriptor rank exceeds four");
  for (int i = 0; i < v.ndim; ++i) {
    v.shape[i] = t.size(i);
    v.strides[i] = t.stride(i);
  }
  switch (t.scalar_type()) {
  case DType::Half: v.dtype = prep::DType::F16; break;
  case DType::BFloat16: v.dtype = prep::DType::BF16; break;
  case DType::Float: v.dtype = prep::DType::F32; break;
  case DType::Int: v.dtype = prep::DType::I32; break;
  case DType::Long: v.dtype = prep::DType::I64; break;
  case DType::Char: v.dtype = prep::DType::I8; break;
  case DType::Bool: v.dtype = prep::DType::Bool; break;
  case DType::Float8_e4m3fn: v.dtype = prep::DType::E4M3; break;
  default: v.dtype = prep::DType::U8; break;
  }
  return v;
}

TensorView attention_view(const Tensor &t) {
  TensorView v;
  v.pointer = t.pointer;
  v.shape.assign(t.shape.begin(), t.shape.end());
  v.device_id = t.device();
  using S = anemoi_sm120::ScalarType;
  switch (t.scalar_type()) {
  case DType::Half: v.dtype = S::Half; break;
  case DType::Float: v.dtype = S::Float; break;
  case DType::Int: v.dtype = S::Int; break;
  case DType::Char: v.dtype = S::Char; break;
  case DType::Float8_e4m3fn: v.dtype = S::Float8_e4m3fn; break;
  case DType::Byte: v.dtype = S::Byte; break;
  default: throw std::invalid_argument("unsupported SM120 attention dtype");
  }
  return v;
}

// Explicit domain-specific recipe, shared by size query and execution. No
// launch, device copy, or device allocation is performed by this constructor.
struct Plan {
  Workspace workspace;
  std::array<Region, 27> prepared;
  std::array<Region, 3> scales, logical_counts, physical_counts;
  Region v_amax, valid_k, probability, max_logits, q_second, k_second;
  Region descriptors, tail_logits, draft_scratch;
  Region logical_ids, precision, keys, sorted_keys, sorted_ids, route_scratch;
  Region physical_ids, video_output, prefix_output;
  std::array<bool, 4> active{};
  std::array<int64_t, 3> budget{}; // FP16, middle, NVFP4
  int64_t prefix_blocks, prefix_qblocks, qtokens, ktokens, kblocks;
  int prefix_phase, first_phase;
  bool native_prefix;
  size_t cub_bytes = 0, draft_bytes = 0;

  explicit Plan(const Config &c) {
    validate(c);
    check(c.architecture == 120 && c.dim == 128,
          "SM120 executor requires architecture 120 and D128");
    check(!c.smooth_k, "SM120 has no native smooth-K specialization");
    prefix_blocks = (int64_t(c.prefix_tokens) + 63) / 64;
    prefix_qblocks = (int64_t(c.prefix_tokens) + c.query_block - 1) / c.query_block;
    qtokens = int64_t(c.video_blocks) * c.query_block;
    ktokens = prefix_blocks * 64 + qtokens;
    kblocks = ktokens / 64;
    prefix_phase = resolved_prefix_kv(c);
    native_prefix = c.prefix_tokens > 0 && resolved_prefix_query(c) == 1 && c.ratios[1] > 0;
    for (int i = 0; i < 4; ++i)
      active[i] = c.ratios[i] > 0 || (c.prefix_tokens > 0 && prefix_phase == i);
    check(!(active[1] && active[2]), "SM120 INT8 and MXFP8 cannot share a route");
    first_phase = 0;
    while (first_phase < 4 && !active[first_phase]) ++first_phase;

    budget = route_budgets(c);
    const int64_t lowest = budget[2] ? budget[2] : budget[1] ? budget[1] : budget[0];
    check(c.anchor_count <= lowest, "anchor_count exceeds lowest-precision budget");

    const int64_t b = c.batch, h = c.heads, r = c.video_blocks;
    const auto add = [&](anemoi_native::IntArrayRef shape, DType type, bool enabled = true) {
      return workspace.add(enabled ? shape : anemoi_native::IntArrayRef{0}, type, c.device);
    };
    const auto half_pool = [&] { return add({b,h,r,128}, DType::Half); };
    prepared[0] = half_pool(); prepared[1] = half_pool();
    prepared[2] = add({b,h,qtokens,128}, DType::Half);
    prepared[3] = add({b,h,ktokens,128}, DType::Half);
    prepared[4] = add({b,h,ktokens,128}, DType::Half);
    // Microscaling Q/K payloads and scales; V is channel-major with physical K64 scales.
    for (int phase : {0, 2}) {
      const bool nv = phase == 0;
      const int base = nv ? 5 : 11;
      const int width = nv ? 64 : 128, scale_width = nv ? 8 : 4;
      prepared[base] = add({b,h,qtokens,width}, DType::Byte, active[phase]);
      prepared[base+1] = add({b,h,qtokens,scale_width}, DType::Byte, active[phase]);
      prepared[base+2] = add({b,h,ktokens,width}, DType::Byte, active[phase]);
      prepared[base+3] = add({b,h,ktokens,scale_width}, DType::Byte, active[phase]);
      prepared[base+4] = add({b,h,128,nv ? ktokens/2 : ktokens}, DType::Byte, active[phase]);
      prepared[base+5] = add({b,h,kblocks,nv ? 512 : 256}, DType::Byte, active[phase]);
    }
    prepared[17] = add({b,h,qtokens,128}, DType::Char, active[1]);
    prepared[18] = add({b,h,r}, DType::Float, active[1]);
    prepared[19] = add({b,h,ktokens,128}, DType::Char, active[1]);
    prepared[20] = add({b,h,kblocks}, DType::Float, active[1]);
    prepared[21] = add({b,h,128,((ktokens+127)/128)*128}, DType::Float8_e4m3fn, active[1]);
    prepared[22] = add({b,h,128}, DType::Float, active[1]);
    prepared[23] = add({b,h,prefix_qblocks*c.query_block,128}, DType::Char, native_prefix);
    prepared[24] = add({b,h,prefix_qblocks}, DType::Float, native_prefix);
    prepared[25] = add({b,h,r,128}, DType::Half, c.maxpool_weight != 0);
    prepared[26] = add({b,h,r,128}, DType::Half, c.maxpool_weight != 0);
    v_amax = add({b,h,kblocks,128}, DType::Float, active[1]);
    for (auto &s : scales) s = add({1}, DType::Float, active[0]);
    valid_k = add({b,kblocks}, DType::Int);
    probability = add({b,h,r,r}, DType::Half);
    max_logits = add({b,h,r,r}, DType::Half, c.maxpool_weight > 0 && c.maxpool_weight < 1);
    q_second = add({b,h,r,128}, DType::Half, c.jensen);
    k_second = add({b,h,r,128}, DType::Half, c.jensen);
    descriptors = add({b,h,r*(c.draft_proxy+1),128}, DType::Half, c.draft_proxy != 0);
    tail_logits = add({b,h,r,r*(c.draft_proxy+1)}, DType::Half, c.draft_proxy != 0);
    if (c.jensen) draft_bytes = draft_workspace_bytes(c.batch,c.heads,c.video_blocks,128,true);
    draft_scratch = add({int64_t(draft_bytes)}, DType::Byte);
    logical_ids = add({b,h,r,r}, DType::Int);
    precision = add({b,h,r,r}, DType::Byte);
    keys = add({b,h,r,r}, DType::Int);
    sorted_keys = add({b,h,r,r}, DType::Int);
    sorted_ids = add({b,h,r,r}, DType::Int);
    for (auto &counts : logical_counts) counts = add({b,h,r}, DType::Int);
    // CUB's exact null-storage query is the only native call in a plan.
    cuda_check(sm120_h3_route_precision_workspace_size(b*h,r,&cub_bytes,nullptr));
    route_scratch = add({int64_t(cub_bytes)}, DType::Byte);
    physical_ids = add({b,h,r,kblocks}, DType::Int);
    physical_counts[0] = add({b,h,r}, DType::Int);
    physical_counts[1] = add({b,h,r}, DType::Int);
    physical_counts[2] = add({b,h,r}, DType::Int, active[3]);
    video_output = add({b,h,qtokens,128}, DType::Half);
    prefix_output = add({b,h,prefix_qblocks*c.query_block,128}, DType::Half, native_prefix);
  }
};
} // namespace

size_t workspace_sm120(const Config &c) { return Plan(c).workspace.bytes; }

void execute_sm120(const Config &c, const Inputs &in, void *workspace,
                   size_t workspace_bytes, cudaStream_t stream) {
  const Plan p(c);
  check(workspace && workspace_bytes >= p.workspace.bytes, "insufficient SM120 workspace");
  const auto get = [&](const Region &r) { return view(r, workspace); };
  std::array<Tensor, 27> t;
  prep::H3Outputs outputs;
  for (int i = 0; i < 27; ++i) {
    t[i] = get(p.prepared[i]);
    outputs[i] = preparation_view(t[i]);
  }
  const Tensor qscale = get(p.scales[0]), kscale = get(p.scales[1]), vscale = get(p.scales[2]);
  if (p.active[0]) fill_scales(c,qscale,kscale,vscale,stream);
  if (p.active[0] || p.active[1] || p.active[2]) {
    prep::H3Options options{c.prefix_tokens,c.query_block,p.active[0],p.active[1],
        p.active[2],p.active[3],p.native_prefix,c.maxpool_weight != 0};
    prep::prepare_h3_sm120_operands(preparation_view(in.q),preparation_view(in.k),
        preparation_view(in.v),preparation_view(in.indices),preparation_view(in.slot_valid),
        preparation_view(in.counts),options,preparation_view(qscale),preparation_view(kscale),
        preparation_view(vscale),outputs,preparation_view(get(p.v_amax)),stream);
  } else {
    pack_h3(c,in,t[2],t[3],t[4],stream);
    pool(t[2],in.counts,c.query_block,t[0],t[25],stream);
    pool(t[3].narrow(2,p.prefix_blocks*64,p.qtokens),in.counts,c.query_block,t[1],t[26],stream);
  }
  const Tensor valid = get(p.valid_k), probability = get(p.probability);
  valid_key_counts(c,in.counts,valid,stream);
  Tensor prefix = in.prefix_output;
  const double scale = 1.0 / std::sqrt(128.0);
  if (p.native_prefix) {
    prefix = get(p.prefix_output);
    const auto function = c.query_block == 64 ? sm120_q64_prefix_int8_attention_forward
                                             : sm120_q128_prefix_int8_attention_forward;
    function(attention_view(t[23]),attention_view(t[19]),attention_view(t[21]),
        attention_view(t[24]),attention_view(t[20]),attention_view(t[22]),
        attention_view(valid),c.prefix_tokens,scale,attention_view(prefix),stream);
    prefix = prefix.narrow(2,0,c.prefix_tokens);
  }
  if (c.jensen) {
    const Tensor qs = get(p.q_second), ks = get(p.k_second);
    moments(t[2],in.counts,c.query_block,qs,stream);
    moments(t[3].narrow(2,p.prefix_blocks*64,p.qtokens),in.counts,c.query_block,ks,stream);
    draft_probability(t[0],t[1],qs,ks,probability,get(p.draft_scratch).pointer,p.draft_bytes,stream);
  } else if (c.draft_proxy) {
    anemoi_sm120::launch_k_tail_probability(t[0].pointer,t[1].pointer,t[3].pointer,
        in.counts.data_ptr<int32_t>(),probability.pointer,get(p.descriptors).pointer,
        get(p.tail_logits).pointer,c.batch,c.heads,c.heads,c.video_blocks,p.prefix_blocks,c.draft_proxy,stream);
  } else {
    anemoi_sm120::launch_draft_probability(t[0].pointer,t[1].pointer,
        t[25].numel() ? t[25].pointer : nullptr,t[26].numel() ? t[26].pointer : nullptr,
        probability.pointer,get(p.max_logits).numel() ? get(p.max_logits).pointer : nullptr,
        c.batch,c.heads,c.heads,c.video_blocks,128,c.maxpool_weight,stream);
  }
  const Tensor logical_ids = get(p.logical_ids), ids = get(p.physical_ids);
  std::array<Tensor, 3> logical, physical;
  for (int i = 0; i < 3; ++i) {
    logical[i] = get(p.logical_counts[i]); physical[i] = get(p.physical_counts[i]);
  }
  cuda_check(sm120_h3_route_precision(probability.pointer,
      c.anchors ? in.anchors.data_ptr<bool>() : nullptr,
      c.anchors ? in.anchor_ids.data_ptr<int>() : nullptr,
      int64_t(c.batch)*c.heads,c.video_blocks,p.budget[0],p.budget[1],p.budget[2],c.anchor_count,
      get(p.precision).data_ptr<uint8_t>(),get(p.keys).data_ptr<uint32_t>(),
      get(p.sorted_keys).data_ptr<uint32_t>(),get(p.sorted_ids).data_ptr<int>(),
      logical_ids.data_ptr<int>(),logical[0].data_ptr<int>(),logical[1].data_ptr<int>(),
      logical[2].data_ptr<int>(),get(p.route_scratch).pointer,p.cub_bytes,stream));
  const int prefix_phase = c.prefix_tokens ? p.prefix_phase : p.first_phase;
  cuda_check(sm120_h3_materialize_route(logical_ids.data_ptr<int>(),logical[0].data_ptr<int>(),
      logical[1].data_ptr<int>(),logical[2].data_ptr<int>(),int64_t(c.batch)*c.heads*c.video_blocks,
      c.video_blocks,c.query_block,p.prefix_blocks,prefix_phase == 0 ? 0 : prefix_phase == 3 ? 2 : 1,
      !c.prefix_tokens || prefix_phase == p.first_phase,p.active[3],ids.data_ptr<int>(),
      physical[0].data_ptr<int>(),physical[1].data_ptr<int>(),
      p.active[3] ? physical[2].data_ptr<int>() : nullptr,stream));

  std::array<TensorView, 27> a;
  for (int i = 0; i < 27; ++i) a[i] = attention_view(t[i]);
  const auto id = attention_view(ids), lo = attention_view(physical[0]),
      mid = attention_view(physical[1]), hi = attention_view(physical[2]), vk = attention_view(valid);
  const auto qg = attention_view(qscale), kg = attention_view(kscale), vg = attention_view(vscale);
  const Tensor video = get(p.video_output);
  const auto out = attention_view(video);
  const int64_t fp16_prefix = p.active[3] && prefix_phase == 3 ? p.prefix_blocks : 0;
  // Each branch selects one native multi-phase kernel; no phase-local output or merge.
#define SM120_SELECT(name) (c.query_block == 64 ? sm120_q64_##name : sm120_q128_##name)
  if (p.active[0] && (p.active[1] || p.active[2])) {
    const int middle = p.active[1] ? 17 : 11;
    const auto function = p.active[1] ? SM120_SELECT(nv_int8_fp16_attention_forward)
                                      : SM120_SELECT(nv_mx_fp16_attention_forward);
    function(a[5],a[6],a[7],a[8],a[9],a[10],a[middle],a[middle+1],a[middle+2],
        a[middle+3],a[middle+4],a[middle+5],a[2],a[3],a[4],id,lo,mid,hi,vk,
        qg,kg,vg,fp16_prefix,scale,p.active[3],out,stream);
  } else if (p.active[0]) {
    SM120_SELECT(nvfp4_attention_forward)(a[5],a[6],a[7],a[8],a[9],a[10],
        a[2],a[3],a[4],id,lo,hi,vk,qg,kg,vg,fp16_prefix,scale,p.active[3],out,stream);
  } else if (p.active[1]) {
    SM120_SELECT(int8_attention_forward)(a[17],a[19],a[21],a[2],a[3],a[4],id,mid,hi,
        a[18],a[20],a[22],vk,fp16_prefix,scale,p.active[3],out,stream);
  } else if (p.active[2]) {
    SM120_SELECT(mxfp8_attention_forward)(a[11],a[12],a[13],a[14],a[15],a[16],
        a[2],a[3],a[4],id,mid,hi,vk,fp16_prefix,scale,p.active[3],out,stream);
  } else {
    SM120_SELECT(fp16_attention_forward)(a[2],a[3],a[4],id,hi,vk,scale,out,stream);
  }
#undef SM120_SELECT
  assemble_h3(c,in,prefix,video,physical[0],physical[1],physical[2],stream);
}
} // namespace comfy::anemoi
