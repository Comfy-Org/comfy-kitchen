/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// 3D Delaunay tetrahedralization by parallel insertion and flipping (gDel3D, Cao et al.
// 2014). The Python driver owns the buffers and the round loop; each kernel is one step:
//
//   vote      each tet holding uninserted points picks one
//   split     1-4 split of each voting tet around its point
//   relocate  points of split tets move to the child containing them
//   detect    non-Delaunay faces of changed tets become 2-3 or 3-2 flip candidates,
//             which claim their tets with atomicMin
//   decide    a candidate runs only if it owns all its tets
//   execute   winning flips rewrite their tets and record where their old faces went
//   fixup     redirect faces pointing at tets flipped in the same pass, set back pointers
//   relocate  points of flipped tets move to the new tet containing them
//
// Tets are positively oriented (orient3d(v0, v1, v2, v3) > 0), face k is opposite vertex
// k, adjacency is (tet << 2 | face) or -1. Predicates use a permanent-based double filter
// and return 0 when undecided; callers treat 0 conservatively (no flip, or "inside" when
// relocating), so the output stays valid even where a degenerate face is non-Delaunay.

#include <cuda_runtime.h>
#include <climits>
#include <cstdint>

#include "utils.cuh"

namespace comfy {
namespace delaunay {

constexpr int kThreads = 256;

__device__ __forceinline__ int enc(int tet, int face) { return (tet << 2) | face; }

// Vertex slots of face k, ordered so that orient3d(face, vertex k) > 0.
__device__ __forceinline__ int face_slot(int k, int l) {
    constexpr int f[4][3] = {{1, 3, 2}, {0, 2, 3}, {0, 3, 1}, {0, 1, 2}};
    return f[k][l];
}

__device__ __forceinline__ const double* vp(const double* P, int v) { return P + 3 * static_cast<int64_t>(v); }

// Double-double (~106 bit) arithmetic for the cases the double filter cannot decide.
struct dd {
    double hi, lo;
};

__device__ __forceinline__ dd quick_two_sum(double a, double b) {
    const double s = a + b;
    return {s, b - (s - a)};
}

__device__ __forceinline__ dd two_sum(double a, double b) {
    const double s = a + b;
    const double bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}

__device__ __forceinline__ dd dd_add(dd a, dd b) {
    const dd s = two_sum(a.hi, b.hi);
    return quick_two_sum(s.hi, s.lo + a.lo + b.lo);
}

__device__ __forceinline__ dd dd_neg(dd a) { return {-a.hi, -a.lo}; }

__device__ __forceinline__ dd dd_mul(dd a, dd b) {
    const double p = a.hi * b.hi;
    return quick_two_sum(p, fma(a.hi, b.hi, -p) + (a.hi * b.lo + a.lo * b.hi));
}

__device__ __noinline__ double orient3d_dd(const double* a, const double* b, const double* c, const double* d) {
    dd u[3], v[3], w[3];  // exact coordinate differences
    for (int k = 0; k < 3; ++k) {
        u[k] = two_sum(b[k], -a[k]);
        v[k] = two_sum(c[k], -a[k]);
        w[k] = two_sum(d[k], -a[k]);
    }
    const dd t1 = dd_add(dd_mul(v[1], w[2]), dd_neg(dd_mul(v[2], w[1])));
    const dd t2 = dd_add(dd_mul(v[2], w[0]), dd_neg(dd_mul(v[0], w[2])));
    const dd t3 = dd_add(dd_mul(v[0], w[1]), dd_neg(dd_mul(v[1], w[0])));
    const dd det = dd_add(dd_add(dd_mul(u[0], t1), dd_mul(u[1], t2)), dd_mul(u[2], t3));
    return det.hi + det.lo;
}

// (b - a) . ((c - a) x (d - a)), or 0 when even double-double cannot separate it from zero.
__device__ __forceinline__ double orient3d(const double* a, const double* b, const double* c, const double* d) {
    const double bx = b[0] - a[0], by = b[1] - a[1], bz = b[2] - a[2];
    const double cx = c[0] - a[0], cy = c[1] - a[1], cz = c[2] - a[2];
    const double dx = d[0] - a[0], dy = d[1] - a[1], dz = d[2] - a[2];
    const double m1 = cy * dz, m2 = cz * dy, m3 = cz * dx, m4 = cx * dz, m5 = cx * dy, m6 = cy * dx;
    const double det = bx * (m1 - m2) + by * (m3 - m4) + bz * (m5 - m6);
    const double perm = fabs(bx) * (fabs(m1) + fabs(m2)) + fabs(by) * (fabs(m3) + fabs(m4)) +
                        fabs(bz) * (fabs(m5) + fabs(m6));
    if (fabs(det) > 1e-14 * perm) return det;
    const double precise = orient3d_dd(a, b, c, d);
    return fabs(precise) > 1e-28 * perm ? precise : 0.0;
}

// Shewchuk's insphere, negated for our orientation: > 0 when e is inside the circumsphere of (a, b, c, d).
__device__ __forceinline__ double insphere(const double* pa, const double* pb, const double* pc, const double* pd,
                                          const double* pe) {
    const double aex = pa[0] - pe[0], aey = pa[1] - pe[1], aez = pa[2] - pe[2];
    const double bex = pb[0] - pe[0], bey = pb[1] - pe[1], bez = pb[2] - pe[2];
    const double cex = pc[0] - pe[0], cey = pc[1] - pe[1], cez = pc[2] - pe[2];
    const double dex = pd[0] - pe[0], dey = pd[1] - pe[1], dez = pd[2] - pe[2];
    const double aexbey = aex * bey, bexaey = bex * aey, bexcey = bex * cey, cexbey = cex * bey;
    const double cexdey = cex * dey, dexcey = dex * cey, dexaey = dex * aey, aexdey = aex * dey;
    const double aexcey = aex * cey, cexaey = cex * aey, bexdey = bex * dey, dexbey = dex * bey;
    const double ab = aexbey - bexaey, bc = bexcey - cexbey, cd = cexdey - dexcey, da = dexaey - aexdey;
    const double ac = aexcey - cexaey, bd = bexdey - dexbey;
    const double abc = aez * bc - bez * ac + cez * ab;
    const double bcd = bez * cd - cez * bd + dez * bc;
    const double cda = cez * da + dez * ac + aez * cd;
    const double dab = dez * ab + aez * bd + bez * da;
    const double alift = aex * aex + aey * aey + aez * aez;
    const double blift = bex * bex + bey * bey + bez * bez;
    const double clift = cex * cex + cey * cey + cez * cez;
    const double dlift = dex * dex + dey * dey + dez * dez;
    const double det = (dlift * abc - clift * dab) + (blift * cda - alift * bcd);
    const double aezp = fabs(aez), bezp = fabs(bez), cezp = fabs(cez), dezp = fabs(dez);
    const double aexbeyp = fabs(aexbey), bexaeyp = fabs(bexaey), bexceyp = fabs(bexcey), cexbeyp = fabs(cexbey);
    const double cexdeyp = fabs(cexdey), dexceyp = fabs(dexcey), dexaeyp = fabs(dexaey), aexdeyp = fabs(aexdey);
    const double aexceyp = fabs(aexcey), cexaeyp = fabs(cexaey), bexdeyp = fabs(bexdey), dexbeyp = fabs(dexbey);
    const double perm =
        ((cexdeyp + dexceyp) * bezp + (dexbeyp + bexdeyp) * cezp + (bexceyp + cexbeyp) * dezp) * alift +
        ((dexaeyp + aexdeyp) * cezp + (aexceyp + cexaeyp) * dezp + (cexdeyp + dexceyp) * aezp) * blift +
        ((aexbeyp + bexaeyp) * dezp + (bexdeyp + dexbeyp) * aezp + (dexaeyp + aexdeyp) * bezp) * clift +
        ((bexceyp + cexbeyp) * aezp + (cexaeyp + aexceyp) * bezp + (aexbeyp + bexaeyp) * cezp) * dlift;
    return fabs(det) > 1e-13 * perm ? -det : 0.0;
}

// Smallest face orientation of q against tet t: >= 0 means inside or undecided.
__device__ __forceinline__ double inside_margin(const int* tet_v, const double* P, int t, const double* q) {
    const int* v = tet_v + 4 * static_cast<int64_t>(t);
    double m = 1.0;
    for (int k = 0; k < 4; ++k) {
        const double o = orient3d(vp(P, v[face_slot(k, 0)]), vp(P, v[face_slot(k, 1)]), vp(P, v[face_slot(k, 2)]), q);
        m = k == 0 ? o : fmin(m, o);
    }
    return m;
}

// Candidate containing q, or the one q is least outside of; candidates < 0 are skipped.
__device__ __forceinline__ int containing_tet(const int* cand, int n_cand, const int* tet_v, const double* P,
                                              const double* q) {
    int best = cand[0];
    double best_margin = -INFINITY;
    for (int k = 0; k < n_cand; ++k) {
        if (cand[k] < 0) continue;
        const double m = inside_margin(tet_v, P, cand[k], q);
        if (m >= 0.0) return cand[k];
        if (m > best_margin) { best_margin = m; best = cand[k]; }
    }
    return best;
}

__device__ __forceinline__ void sort3(int& a, int& b, int& c) {
    if (a > b) { const int t = a; a = b; b = t; }
    if (b > c) { const int t = b; b = c; c = t; }
    if (a > b) { const int t = a; a = b; b = t; }
}

__device__ __forceinline__ void face_key(const int* v, int k, int& a, int& b, int& c) {
    a = v[face_slot(k, 0)];
    b = v[face_slot(k, 1)];
    c = v[face_slot(k, 2)];
    sort3(a, b, c);
}

__global__ void vote_kernel(const int* __restrict__ pt_tet, const int* __restrict__ pt_prio, int n_pts,
                            int* __restrict__ vote) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_pts) return;
    const int t = pt_tet[q];
    if (t >= 0) atomicMin(&vote[t], pt_prio[q]);
}

// Child k of a split tet replaces vertex k with the new point; child 0 keeps the slot.
__device__ __forceinline__ int split_child(int t, int k, int base_new, const int* split_rank) {
    return k == 0 ? t : base_new + 3 * split_rank[t] + k - 1;
}

__global__ void split_kernel(int n_tets, const int* __restrict__ vote, const int* __restrict__ split_rank,
                             const int* __restrict__ prio_to_pt, int* tet_v, int* tet_opp,
                             unsigned char* alive, int* pt_tet, unsigned char* active) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tets || vote[t] == INT_MAX) return;
    const int p = prio_to_pt[vote[t]];
    int v[4], o[4], child[4];
    for (int k = 0; k < 4; ++k) {
        v[k] = tet_v[4 * t + k];
        o[k] = tet_opp[4 * t + k];
        child[k] = split_child(t, k, n_tets, split_rank);
    }
    for (int k = 0; k < 4; ++k) {
        const int c = child[k];
        for (int l = 0; l < 4; ++l) tet_v[4 * c + l] = l == k ? p : v[l];
        for (int j = 0; j < 4; ++j)
            if (j != k) tet_opp[4 * c + j] = enc(child[j], k);
        if (o[k] < 0) {
            tet_opp[4 * c + k] = -1;
        } else {
            const int nb = o[k] >> 2, f = o[k] & 3;
            if (vote[nb] != INT_MAX) {
                tet_opp[4 * c + k] = enc(split_child(nb, f, n_tets, split_rank), f);
            } else {
                tet_opp[4 * c + k] = o[k];
                tet_opp[4 * nb + f] = enc(c, k);
            }
        }
        alive[c] = 1;
        active[c] = 1;
    }
    pt_tet[p] = -1;
}

__global__ void relocate_split_kernel(int n_pts, const int* __restrict__ vote, const int* __restrict__ split_rank,
                                      int base_new, const int* __restrict__ tet_v, const double* __restrict__ P,
                                      int* pt_tet) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_pts) return;
    const int t = pt_tet[q];
    if (t < 0 || vote[t] == INT_MAX) return;
    int child[4];
    for (int k = 0; k < 4; ++k) child[k] = split_child(t, k, base_new, split_rank);
    pt_tet[q] = containing_tet(child, 4, tet_v, P, vp(P, q));
}

// cand_info per candidate: 0 none, 1 = 2-3 flip, 2 | (reflex edge << 2) = 3-2 flip.
__global__ void flip_detect_kernel(int n_active, const int* __restrict__ active_list,
                                   const unsigned char* __restrict__ active_flag, const int* __restrict__ tet_v,
                                   const int* __restrict__ tet_opp, const double* __restrict__ P, int* cand_info,
                                   int* cand_n, int* cand_m, int* owner, unsigned char* retry) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= n_active) return;
    const int t = active_list[a];
    const int* tv = tet_v + 4 * t;
    for (int i = 0; i < 4; ++i) {
        const int cid = 4 * a + i;
        cand_info[cid] = 0;
        const int o = tet_opp[4 * t + i];
        if (o < 0) continue;
        const int n = o >> 2, f = o & 3;
        if (active_flag[n] && n < t) continue;  // n checks this face
        const int d = tv[i], e = tet_v[4 * n + f];
        if (insphere(vp(P, tv[0]), vp(P, tv[1]), vp(P, tv[2]), vp(P, tv[3]), vp(P, e)) <= 0.0) continue;
        int x[3];
        for (int l = 0; l < 3; ++l) x[l] = tv[face_slot(i, l)];
        // an edge whose 2-3 tet (x_k, x_k+1, e, d) would invert is reflex; only a 3-2 flip removes it
        int reflex = -1, n_reflex = 0;
        bool unsure = false;
        for (int k = 0; k < 3; ++k) {
            const double s = orient3d(vp(P, x[k]), vp(P, x[(k + 1) % 3]), vp(P, e), vp(P, d));
            if (s == 0.0) unsure = true;
            else if (s < 0.0) { reflex = k; ++n_reflex; }
        }
        if (unsure || n_reflex > 1) {  // may become flippable once its neighbours flip
            retry[t] = 1;
            continue;
        }
        int m = -1, info = 1;
        if (n_reflex == 1) {
            const int xa = x[reflex], xb = x[(reflex + 1) % 3], xc = x[(reflex + 2) % 3];
            int it = 0, in = 0;
            for (int l = 0; l < 4; ++l) {
                if (tv[l] == xc) it = l;
                if (tet_v[4 * n + l] == xc) in = l;
            }
            const int m1 = tet_opp[4 * t + it], m2 = tet_opp[4 * n + in];
            const double oa = orient3d(vp(P, xc), vp(P, d), vp(P, e), vp(P, xa));
            const double ob = orient3d(vp(P, xc), vp(P, d), vp(P, e), vp(P, xb));
            // xa-xb must be shared by exactly these 3 tets and the 2 new ones must not invert
            if (m1 < 0 || m2 < 0 || (m1 >> 2) != (m2 >> 2) || oa == 0.0 || ob == 0.0 || (oa > 0.0) == (ob > 0.0)) {
                retry[t] = 1;
                continue;
            }
            m = m1 >> 2;
            info = 2 | (reflex << 2);
        }
        cand_info[cid] = info;
        cand_n[cid] = n;
        cand_m[cid] = m;
        atomicMin(&owner[t], cid);
        atomicMin(&owner[n], cid);
        if (m >= 0) atomicMin(&owner[m], cid);
    }
}

__global__ void flip_decide_kernel(int n_cand, const int* __restrict__ active_list, const int* __restrict__ cand_info,
                                   const int* __restrict__ cand_n, const int* __restrict__ cand_m,
                                   const int* __restrict__ owner, int* exec, unsigned char* active_next) {
    const int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= n_cand) return;
    const int info = cand_info[cid];
    exec[cid] = 0;
    if (!info) return;
    const int t = active_list[cid >> 2], n = cand_n[cid], m = cand_m[cid];
    if (owner[t] == cid && owner[n] == cid && (m < 0 || owner[m] == cid)) {
        exec[cid] = info;
    } else {  // retry next pass
        active_next[t] = 1;
        active_next[n] = 1;
    }
}

// run_rank is the flip's rank among those running this pass: new_tets is indexed by it,
// flipped[old tet] holds it + 1.
__global__ void flip_execute_kernel(int n_cand, const int* __restrict__ active_list, const int* __restrict__ exec,
                                    const int* __restrict__ cand_n, const int* __restrict__ cand_m,
                                    const int* __restrict__ slot_rank, const int* __restrict__ run_rank,
                                    int base_new, const double* __restrict__ P,
                                    int* tet_v, int* tet_opp, unsigned char* alive, int* flipped, int* redirect,
                                    unsigned char* pend, int* new_tets, unsigned char* active_next) {
    const int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if (cid >= n_cand) return;
    const int info = exec[cid];
    if (!info) return;
    const int i = cid & 3;
    const int old_slot[3] = {active_list[cid >> 2], cand_n[cid], cand_m[cid]};
    const int n_old = (info & 3) == 1 ? 2 : 3;
    int ov[3][4], oo[3][4];
    for (int o = 0; o < n_old; ++o)
        for (int l = 0; l < 4; ++l) {
            ov[o][l] = tet_v[4 * old_slot[o] + l];
            oo[o][l] = tet_opp[4 * old_slot[o] + l];
        }
    const int d = ov[0][i];
    const int e = ov[1][oo[0][i] & 3];
    int x[3];
    for (int l = 0; l < 3; ++l) x[l] = ov[0][face_slot(i, l)];

    int nv[3][4], slot[3], n_new;
    if ((info & 3) == 1) {
        n_new = 3;
        for (int k = 0; k < 3; ++k) {
            nv[k][0] = x[k];
            nv[k][1] = x[(k + 1) % 3];
            nv[k][2] = e;
            nv[k][3] = d;
        }
        slot[0] = old_slot[0];
        slot[1] = old_slot[1];
        slot[2] = base_new + slot_rank[cid];
    } else {
        n_new = 2;
        const int r = info >> 2;
        const int xc = x[(r + 2) % 3];
        for (int k = 0; k < 2; ++k) {
            const int apex = x[(r + k) % 3];
            nv[k][0] = xc;
            nv[k][1] = d;
            nv[k][2] = e;
            nv[k][3] = apex;
            if (orient3d(vp(P, xc), vp(P, d), vp(P, e), vp(P, apex)) < 0.0) {
                nv[k][1] = e;
                nv[k][2] = d;
            }
        }
        slot[0] = old_slot[0];
        slot[1] = old_slot[1];
        slot[2] = -1;
        alive[old_slot[2]] = 0;
    }

    for (int a = 0; a < n_new; ++a) {
        for (int j = 0; j < 4; ++j) {
            int fa, fb, fc;
            face_key(nv[a], j, fa, fb, fc);
            bool outer = false;  // face of an old tet: keeps its neighbour, fixed up later
            for (int o = 0; o < n_old && !outer; ++o)
                for (int g = 0; g < 4; ++g) {
                    int ga, gb, gc;
                    face_key(ov[o], g, ga, gb, gc);
                    if (ga == fa && gb == fb && gc == fc) {
                        tet_opp[4 * slot[a] + j] = oo[o][g];
                        pend[4 * slot[a] + j] = oo[o][g] >= 0;
                        redirect[4 * old_slot[o] + g] = enc(slot[a], j);
                        outer = true;
                        break;
                    }
                }
            if (outer) continue;
            for (int b = 0; b < n_new; ++b) {  // face shared with another new tet
                if (b == a) continue;
                for (int h = 0; h < 4; ++h) {
                    int ha, hb, hc;
                    face_key(nv[b], h, ha, hb, hc);
                    if (ha == fa && hb == fb && hc == fc) {
                        tet_opp[4 * slot[a] + j] = enc(slot[b], h);
                        pend[4 * slot[a] + j] = 0;
                    }
                }
            }
        }
    }
    for (int a = 0; a < n_new; ++a) {
        for (int l = 0; l < 4; ++l) tet_v[4 * slot[a] + l] = nv[a][l];
        alive[slot[a]] = 1;
        active_next[slot[a]] = 1;
    }
    const int r = run_rank[cid];
    for (int o = 0; o < n_old; ++o) flipped[old_slot[o]] = r + 1;
    for (int k = 0; k < 3; ++k) new_tets[3 * r + k] = slot[k];
}

__global__ void flip_fixup_kernel(int n_run, const int* __restrict__ new_tets, const int* __restrict__ flipped,
                                  const int* __restrict__ redirect, int* tet_opp, unsigned char* pend) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_run) return;
    for (int k = 0; k < 3; ++k) {
        const int s = new_tets[3 * r + k];
        if (s < 0) continue;
        for (int j = 0; j < 4; ++j) {
            if (!pend[4 * s + j]) continue;
            pend[4 * s + j] = 0;
            const int x = tet_opp[4 * s + j];
            const int X = x >> 2, g = x & 3;
            if (flipped[X]) tet_opp[4 * s + j] = redirect[4 * X + g];  // that neighbour was replaced this pass too
            else tet_opp[4 * X + g] = enc(s, j);
        }
    }
}

__global__ void relocate_flip_kernel(int n_pts, const int* __restrict__ flipped, const int* __restrict__ new_tets,
                                     const int* __restrict__ tet_v, const double* __restrict__ P, int* pt_tet) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= n_pts) return;
    const int s = pt_tet[q];
    if (s < 0 || !flipped[s]) return;
    pt_tet[q] = containing_tet(new_tets + 3 * (flipped[s] - 1), 3, tet_v, P, vp(P, q));
}

__global__ void compact_kernel(int n_tets, const unsigned char* __restrict__ alive, const int* __restrict__ new_index,
                               const int* __restrict__ tet_v, const int* __restrict__ tet_opp, int* out_tets,
                               int* out_nbr) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tets || !alive[t]) return;
    const int r = new_index[t];
    for (int k = 0; k < 4; ++k) {
        out_tets[4 * r + k] = tet_v[4 * t + k];
        const int o = tet_opp[4 * t + k];
        out_nbr[4 * r + k] = o < 0 ? -1 : enc(new_index[o >> 2], o & 3);
    }
}

inline unsigned int blocks_for(int n) { return static_cast<unsigned int>((n + kThreads - 1) / kThreads); }


}  // namespace delaunay
}  // namespace comfy

using namespace comfy::delaunay;

extern "C" {

void launch_delaunay_vote(const void* pt_tet, const void* pt_prio, int n_pts, void* vote, cudaStream_t stream) {
    if (n_pts <= 0) return;
    vote_kernel<<<blocks_for(n_pts), kThreads, 0, stream>>>(
        static_cast<const int*>(pt_tet), static_cast<const int*>(pt_prio), n_pts, static_cast<int*>(vote));
    CUDA_CHECK(cudaGetLastError());
}

void launch_delaunay_split(int n_tets, const void* vote, const void* split_rank, const void* prio_to_pt, void* tet_v,
                           void* tet_opp, void* alive, void* pt_tet, void* active, cudaStream_t stream) {
    if (n_tets <= 0) return;
    split_kernel<<<blocks_for(n_tets), kThreads, 0, stream>>>(
        n_tets, static_cast<const int*>(vote), static_cast<const int*>(split_rank),
        static_cast<const int*>(prio_to_pt), static_cast<int*>(tet_v), static_cast<int*>(tet_opp),
        static_cast<unsigned char*>(alive), static_cast<int*>(pt_tet), static_cast<unsigned char*>(active));
    CUDA_CHECK(cudaGetLastError());
}

void launch_delaunay_relocate_split(int n_pts, const void* vote, const void* split_rank, int base_new,
                                    const void* tet_v, const void* points, void* pt_tet, cudaStream_t stream) {
    if (n_pts <= 0) return;
    relocate_split_kernel<<<blocks_for(n_pts), kThreads, 0, stream>>>(
        n_pts, static_cast<const int*>(vote), static_cast<const int*>(split_rank), base_new,
        static_cast<const int*>(tet_v), static_cast<const double*>(points), static_cast<int*>(pt_tet));
    CUDA_CHECK(cudaGetLastError());
}

void launch_delaunay_flip_detect(int n_active, const void* active_list, const void* active_flag, const void* tet_v,
                                 const void* tet_opp, const void* points, void* cand_info, void* cand_n, void* cand_m,
                                 void* owner, void* retry, cudaStream_t stream) {
    if (n_active <= 0) return;
    flip_detect_kernel<<<blocks_for(n_active), kThreads, 0, stream>>>(
        n_active, static_cast<const int*>(active_list), static_cast<const unsigned char*>(active_flag),
        static_cast<const int*>(tet_v), static_cast<const int*>(tet_opp), static_cast<const double*>(points),
        static_cast<int*>(cand_info), static_cast<int*>(cand_n), static_cast<int*>(cand_m), static_cast<int*>(owner),
        static_cast<unsigned char*>(retry));
    CUDA_CHECK(cudaGetLastError());
}

void launch_delaunay_flip_decide(int n_cand, const void* active_list, const void* cand_info, const void* cand_n,
                                 const void* cand_m, const void* owner, void* exec, void* active_next,
                                 cudaStream_t stream) {
    if (n_cand <= 0) return;
    flip_decide_kernel<<<blocks_for(n_cand), kThreads, 0, stream>>>(
        n_cand, static_cast<const int*>(active_list), static_cast<const int*>(cand_info),
        static_cast<const int*>(cand_n), static_cast<const int*>(cand_m), static_cast<const int*>(owner),
        static_cast<int*>(exec), static_cast<unsigned char*>(active_next));
    CUDA_CHECK(cudaGetLastError());
}

void launch_delaunay_flip_execute(int n_cand, const void* active_list, const void* exec, const void* cand_n,
                                  const void* cand_m, const void* slot_rank, const void* run_rank, int base_new,
                                  const void* points,
                                  void* tet_v, void* tet_opp, void* alive, void* flipped, void* redirect, void* pend,
                                  void* new_tets, void* active_next, cudaStream_t stream) {
    if (n_cand <= 0) return;
    flip_execute_kernel<<<blocks_for(n_cand), kThreads, 0, stream>>>(
        n_cand, static_cast<const int*>(active_list), static_cast<const int*>(exec), static_cast<const int*>(cand_n),
        static_cast<const int*>(cand_m), static_cast<const int*>(slot_rank), static_cast<const int*>(run_rank),
        base_new, static_cast<const double*>(points), static_cast<int*>(tet_v), static_cast<int*>(tet_opp),
        static_cast<unsigned char*>(alive), static_cast<int*>(flipped), static_cast<int*>(redirect),
        static_cast<unsigned char*>(pend), static_cast<int*>(new_tets), static_cast<unsigned char*>(active_next));
    CUDA_CHECK(cudaGetLastError());
}

void launch_delaunay_flip_fixup(int n_run, const void* new_tets, const void* flipped, const void* redirect,
                                void* tet_opp, void* pend, cudaStream_t stream) {
    if (n_run <= 0) return;
    flip_fixup_kernel<<<blocks_for(n_run), kThreads, 0, stream>>>(
        n_run, static_cast<const int*>(new_tets), static_cast<const int*>(flipped),
        static_cast<const int*>(redirect), static_cast<int*>(tet_opp), static_cast<unsigned char*>(pend));
    CUDA_CHECK(cudaGetLastError());
}

void launch_delaunay_relocate_flip(int n_pts, const void* flipped, const void* new_tets, const void* tet_v,
                                   const void* points, void* pt_tet, cudaStream_t stream) {
    if (n_pts <= 0) return;
    relocate_flip_kernel<<<blocks_for(n_pts), kThreads, 0, stream>>>(
        n_pts, static_cast<const int*>(flipped), static_cast<const int*>(new_tets), static_cast<const int*>(tet_v),
        static_cast<const double*>(points), static_cast<int*>(pt_tet));
    CUDA_CHECK(cudaGetLastError());
}

void launch_delaunay_compact(int n_tets, const void* alive, const void* new_index, const void* tet_v,
                             const void* tet_opp, void* out_tets, void* out_nbr, cudaStream_t stream) {
    if (n_tets <= 0) return;
    compact_kernel<<<blocks_for(n_tets), kThreads, 0, stream>>>(
        n_tets, static_cast<const unsigned char*>(alive), static_cast<const int*>(new_index),
        static_cast<const int*>(tet_v), static_cast<const int*>(tet_opp), static_cast<int*>(out_tets),
        static_cast<int*>(out_nbr));
    CUDA_CHECK(cudaGetLastError());
}

}  // extern "C"
