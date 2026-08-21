"""Sub-pooling tau sweep on a captured H3 attention call.

Answers, on REAL latents, whether a finer pooled tail (sub>1) at high tau beats
the current sub=1 at low tau: prints exact-block density and cosine-vs-dense for
a tau x sub grid. Routing stays at 64-token blocks for every sub; only the
pooled tail granularity changes.

Capture first: run ComfyUI with SOL_ATTN_CAPTURE=/path/h3.pt (see
sol_attn_minimax.py), then:

    python samples/sol_subpool_sweep.py /path/h3.pt
"""
import argparse

import torch
import torch.nn.functional as F

BLOCK = 64
LOG2E = 1.4426950408889634
NEG = float("-inf")


def pool(x, chunk, t, reduce):
    """(T, D) fp32 -> (ceil/chunk, D) mean or sum over real tokens only."""
    n = (t + chunk - 1) // chunk
    pad = n * chunk - t
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    blocks = x.view(n, chunk, -1)
    lengths = torch.full((n,), float(chunk), device=x.device)
    if pad:
        lengths[-1] = float(chunk - pad)
    s = blocks.sum(dim=1)
    if reduce == "sum":
        return s, lengths
    return s / lengths.clamp_min(1.0).unsqueeze(-1), lengths


def sweep_head(qh, kh, vh, scale, tau, sub, sink, sink_q, chunk_rows):
    """One (head, tau, sub) config. All fp32, chunked over query rows.

    Returns (out [T, D], exact-density). Mirrors the eager reference: centred
    K, exp2 softmax, centroid tail shared per query block, tail terms carry
    their real token count in the denominator.
    """
    t, d = qh.shape
    n = (t + BLOCK - 1) // BLOCK
    log2s = scale * LOG2E

    kc64, _ = pool(kh, BLOCK, t, "mean")
    k_mean = kc64.mean(dim=0, keepdim=True)
    kcc64 = kc64 - k_mean
    khc = kh - k_mean
    centroid, _ = pool(qh, BLOCK, t, "mean")

    kc_var = kcc64.pow(2).mean(dim=0)
    var = (centroid.pow(2) * kc_var.unsqueeze(0)).sum(-1)
    thr = tau * torch.sqrt(var * log2s * log2s + 1e-6)

    colmean = (centroid @ kcc64.T) * log2s                        # (NQ, N)
    idx = torch.arange(n, device=qh.device)
    exact = colmean > thr.unsqueeze(-1)
    exact |= (idx.view(1, -1) - idx.view(-1, 1)).abs() <= 1
    exact |= ((idx >= sink[0]) & (idx < sink[1])).view(1, -1)
    exact |= ((idx >= sink_q[0]) & (idx < sink_q[1])).view(-1, 1)
    valid = (idx * BLOCK < t).view(1, -1)
    exact &= valid
    density = exact[valid.expand_as(exact)].float().mean().item()

    sub_chunk = BLOCK // sub
    kc_s, len_s = pool(khc, sub_chunk, t, "mean")                 # (N*sub, D)
    vc_s, _ = pool(vh, sub_chunk, t, "sum")
    s_tail = (centroid @ kc_s.T) * log2s                          # (NQ, N*sub)
    tail_block = torch.arange(kc_s.shape[0], device=qh.device) // sub
    dead = exact.gather(1, tail_block.view(1, -1).expand(n, -1))
    dead |= (len_s <= 0).view(1, -1)
    s_tail = s_tail.masked_fill(dead, NEG)

    out = torch.empty(t, d, device=qh.device)
    cols = torch.arange(t, device=qh.device) // BLOCK
    for r0 in range(0, t, chunk_rows):
        r1 = min(r0 + chunk_rows, t)
        qb = cols[r0:r1]
        s_tok = (qh[r0:r1] @ khc.T) * log2s                       # (R, T)
        keep = exact[qb][:, cols]
        s_tok = s_tok.masked_fill(~keep, NEG)
        s_tl = s_tail[qb]                                         # (R, N*sub)
        m = torch.maximum(s_tok.amax(-1), s_tl.amax(-1)).unsqueeze(-1)
        p_tok = torch.exp2(s_tok - m)
        p_tl = torch.exp2(s_tl - m)
        p_tok[~keep] = 0.0
        p_tl[s_tl == NEG] = 0.0
        num = p_tok @ vh + p_tl @ vc_s
        den = p_tok.sum(-1) + (p_tl * len_s.unsqueeze(0)).sum(-1)
        out[r0:r1] = num / den.clamp_min(1e-30).unsqueeze(-1)
    return out, density


def flat_cos(a, b):
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--taus", default="1.4,2.0,3.0,4.0,6.0")
    ap.add_argument("--subs", default="1,2,4,8")
    ap.add_argument("--heads", type=int, default=0, help="0 = all")
    ap.add_argument("--chunk-rows", type=int, default=2048)
    args = ap.parse_args()

    d = torch.load(args.capture, map_location="cuda", weights_only=False)
    q, k, v = d["q"].cuda(), d["k"].cuda(), d["v"].cuda()        # (B, T, H, D)
    b, t, h, hd = q.shape
    assert b == 1, "capture is per-call, expected batch 1"
    scale = d["scale"] or hd ** -0.5
    sink, sink_q = d["sink_blocks"], d["sink_q"]
    vspan = d.get("video_span") or (sink[1] * BLOCK, t)
    heads = list(range(h)) if not args.heads else \
        list(range(0, h, max(1, h // args.heads)))[: args.heads]
    print(f"T={t} H={h} sink={sink} sink_q={sink_q} video={vspan} "
          f"heads={len(heads)}")

    # Dense reference: flash, fp32 accumulation internally.
    ref = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3),
        scale=scale)[0].float()                                   # (H, T, D)

    taus = [float(x) for x in args.taus.split(",")]
    subs = [int(x) for x in args.subs.split(",")]
    print(f"{'tau':>5} {'sub':>4} {'density':>8} {'cos_all':>9} {'cos_video':>9}")
    for tau in taus:
        for sub in subs:
            outs, dens = [], []
            for hi in heads:
                qh = q[0, :, hi].float()
                o, dy = sweep_head(qh, k[0, :, hi].float(), v[0, :, hi].float(),
                                   scale, tau, sub, sink, sink_q,
                                   args.chunk_rows)
                outs.append(o)
                dens.append(dy)
            out = torch.stack(outs)
            r = ref[heads]
            cos_all = flat_cos(out, r)
            cos_vid = flat_cos(out[:, vspan[0]:vspan[1]],
                               r[:, vspan[0]:vspan[1]]) if vspan else cos_all
            print(f"{tau:5.1f} {sub:4d} {sum(dens)/len(dens):8.4f} "
                  f"{cos_all:9.5f} {cos_vid:9.5f}", flush=True)


if __name__ == "__main__":
    main()
