// SPDX-License-Identifier: Apache-2.0
// Geometry ops: Delaunay, triangle BVH queries, connected components, edge-collapse
// checks and min cut. None of these kernels depend on the wavefront size, so this unit
// is linked into _C (wave32, RDNA) and into _geom (native wave64, CDNA).
#include <cstdint>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "binding_checks.h"

extern "C" {
// 3D Delaunay steps, looped from Python, see ops/delaunay3d.hip.
void launch_delaunay_vote(const void* pt_tet, const void* pt_prio, int n_pts, void* vote, hipStream_t stream);
void launch_delaunay_split(int n_tets, const void* vote, const void* split_rank, const void* prio_to_pt,
                           void* tet_v, void* tet_opp, void* alive, void* pt_tet, void* active, hipStream_t stream);
void launch_delaunay_relocate_split(int n_pts, const void* vote, const void* split_rank, int base_new,
                                    const void* tet_v, const void* points, void* pt_tet, hipStream_t stream);
void launch_delaunay_flip_detect(int n_active, const void* active_list, const void* active_flag,
                                 const void* tet_v, const void* tet_opp, const void* points, void* cand_info,
                                 void* cand_n, void* cand_m, void* owner, void* retry, hipStream_t stream);
void launch_delaunay_flip_decide(int n_cand, const void* active_list, const void* cand_info, const void* cand_n,
                                 const void* cand_m, const void* owner, void* exec, void* active_next,
                                 hipStream_t stream);
void launch_delaunay_flip_execute(int n_cand, const void* active_list, const void* exec, const void* cand_n,
                                  const void* cand_m, const void* slot_rank, const void* run_rank, int base_new,
                                  const void* points, void* tet_v, void* tet_opp, void* alive, void* flipped,
                                  void* redirect, void* pend, void* new_tets, void* active_next,
                                  hipStream_t stream);
void launch_delaunay_flip_fixup(int n_run, const void* new_tets, const void* flipped, const void* redirect,
                                void* tet_opp, void* pend, hipStream_t stream);
void launch_delaunay_relocate_flip(int n_pts, const void* flipped, const void* new_tets, const void* tet_v,
                                   const void* points, void* pt_tet, hipStream_t stream);
void launch_delaunay_compact(int n_tets, const void* alive, const void* new_index, const void* tet_v,
                             const void* tet_opp, void* out_tets, void* out_nbr, hipStream_t stream);

// Triangle BVH and closest-point queries, see ops/mesh_bvh.hip.
void launch_mesh_bvh_build(const void* codes, const void* tris, int n_leaves, void* child, void* parent,
                           void* visits, void* box, hipStream_t stream);
void launch_closest_point_on_mesh(const void* points, int n_points, const void* tris, const void* tri_index,
                                  int n_leaves, const void* child, const void* box, float max_dist, void* dist,
                                  void* closest, void* face, hipStream_t stream);
void launch_closest_hit_on_mesh(const void* origins, const void* directions, int n_rays, const void* tris,
                                const void* tri_index, int n_leaves, const void* child, const void* box, float tmin,
                                float tmax, bool cull_backface, void* t, void* face, hipStream_t stream);

// Union-find connected components, see ops/connected_components.hip.
void launch_connected_components(const void* edges, int64_t n_edges, bool edges_int64, int n_nodes,
                                 void* parent, void* labels, hipStream_t stream);

// Edge-collapse checks, see ops/edge_collapse.hip.
void launch_edge_collapse_checks(const void* verts, const void* faces, const void* offsets, void* cursor, void* fan,
                                 const void* edges, const void* positions, int n_corners, int n_edges,
                                 float cos_threshold, void* flips, void* skinny, void* link_ok, hipStream_t stream);

// Push-relabel min cut on bounded-degree graphs, see ops/min_cut.hip.
void launch_min_cut_init(int n, int K, const void* nbr, const void* cap, void* r, void* rev, hipStream_t stream);
void launch_min_cut_global_relabel(int n, int K, const void* nbr, const void* rev, const void* r, const void* rt,
                                   float tol, int big, void* h, void* lists, void* counts, hipStream_t stream);
void launch_min_cut_rebuild_active(int n, const void* e, const void* h, float tol, int big, void* list, void* count,
                                   void* queued, hipStream_t stream);
void launch_min_cut_iterate(int n, int K, const void* nbr, const void* rev, void* r, void* e, void* rt, void* h,
                            void* inc, void* hn, float tol, int big, const void* list, const void* count,
                            void* list_out, void* count_out, void* queued, void* active, hipStream_t stream);
}

// Geometry ops. The buffers are owned by Python; the arrays are device-typed, so a host
// operand is rejected before its pointer reaches a kernel.
using rocm_array = nb::ndarray<nb::device::rocm>;

static constexpr auto kInt = nb::dlpack::dtype_code::Int;
static constexpr auto kUInt = nb::dlpack::dtype_code::UInt;
static constexpr auto kFloat = nb::dlpack::dtype_code::Float;
static constexpr auto kBool = nb::dlpack::dtype_code::Bool;

// Negative, so require_len rejects it, when either factor is negative.
static int64_t extent(int64_t a, int64_t b) { return a < 0 || b < 0 ? -1 : a * b; }

// need bounds what the launch extents determine; a buffer indexed through another buffer's values passes 0,
// and those values are checked in the Python wrappers.
static void require_buf(const rocm_array& a, nb::dlpack::dtype_code code, int bits, int64_t need, const char* fn,
                        const char* name) {
    const nb::ndarray<> t(a);
    if (t.dtype().code != static_cast<uint8_t>(code) || t.dtype().bits != bits) {
        throw std::runtime_error(std::string(fn) + ": " + name + " has the wrong dtype, expected " +
                                 std::to_string(bits) + "-bit");
    }
    require_len(t, need, fn, name);
    if (t.size() != 0) require_packed_contiguous(t, fn, name);  // an empty array's strides mean nothing
}

void delaunay_vote(rocm_array pt_tet, rocm_array pt_prio, int n_pts, rocm_array vote, uintptr_t stream_ptr) {
    constexpr const char* kFn = "delaunay_vote";
    require_buf(pt_tet, kInt, 32, n_pts, kFn, "pt_tet");
    require_buf(pt_prio, kInt, 32, n_pts, kFn, "pt_prio");
    require_buf(vote, kInt, 32, 0, kFn, "vote");
    launch_delaunay_vote(pt_tet.data(), pt_prio.data(), n_pts, vote.data(),
                         reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void delaunay_split(int n_tets, rocm_array vote, rocm_array split_rank, rocm_array prio_to_pt, rocm_array tet_v,
                    rocm_array tet_opp, rocm_array alive, rocm_array pt_tet, rocm_array active, uintptr_t stream_ptr) {
    constexpr const char* kFn = "delaunay_split";
    require_buf(vote, kInt, 32, n_tets, kFn, "vote");
    require_buf(split_rank, kInt, 32, n_tets, kFn, "split_rank");
    require_buf(prio_to_pt, kInt, 32, 0, kFn, "prio_to_pt");
    require_buf(tet_v, kInt, 32, extent(4, n_tets), kFn, "tet_v");
    require_buf(tet_opp, kInt, 32, extent(4, n_tets), kFn, "tet_opp");
    require_buf(alive, kUInt, 8, n_tets, kFn, "alive");
    require_buf(pt_tet, kInt, 32, 0, kFn, "pt_tet");
    require_buf(active, kUInt, 8, n_tets, kFn, "active");
    launch_delaunay_split(n_tets, vote.data(), split_rank.data(), prio_to_pt.data(), tet_v.data(), tet_opp.data(),
                          alive.data(), pt_tet.data(), active.data(), reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void delaunay_relocate_split(int n_pts, rocm_array vote, rocm_array split_rank, int base_new, rocm_array tet_v,
                             rocm_array points, rocm_array pt_tet, uintptr_t stream_ptr) {
    constexpr const char* kFn = "delaunay_relocate_split";
    require_buf(vote, kInt, 32, 0, kFn, "vote");
    require_buf(split_rank, kInt, 32, 0, kFn, "split_rank");
    require_buf(tet_v, kInt, 32, extent(4, base_new), kFn, "tet_v");
    require_buf(points, kFloat, 64, extent(3, n_pts), kFn, "points");
    require_buf(pt_tet, kInt, 32, n_pts, kFn, "pt_tet");
    launch_delaunay_relocate_split(n_pts, vote.data(), split_rank.data(), base_new, tet_v.data(), points.data(),
                                   pt_tet.data(), reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void delaunay_flip_detect(int n_active, rocm_array active_list, rocm_array active_flag, rocm_array tet_v,
                          rocm_array tet_opp, rocm_array points, rocm_array cand_info, rocm_array cand_n,
                          rocm_array cand_m, rocm_array owner, rocm_array retry, uintptr_t stream_ptr) {
    constexpr const char* kFn = "delaunay_flip_detect";
    require_buf(active_list, kInt, 32, n_active, kFn, "active_list");
    require_buf(active_flag, kUInt, 8, 0, kFn, "active_flag");
    require_buf(tet_v, kInt, 32, 0, kFn, "tet_v");
    require_buf(tet_opp, kInt, 32, 0, kFn, "tet_opp");
    require_buf(points, kFloat, 64, 0, kFn, "points");
    require_buf(cand_info, kInt, 32, extent(4, n_active), kFn, "cand_info");
    require_buf(cand_n, kInt, 32, extent(4, n_active), kFn, "cand_n");
    require_buf(cand_m, kInt, 32, extent(4, n_active), kFn, "cand_m");
    require_buf(owner, kInt, 32, 0, kFn, "owner");
    require_buf(retry, kUInt, 8, 0, kFn, "retry");
    launch_delaunay_flip_detect(n_active, active_list.data(), active_flag.data(), tet_v.data(), tet_opp.data(),
                                points.data(), cand_info.data(), cand_n.data(), cand_m.data(), owner.data(),
                                retry.data(), reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void delaunay_flip_decide(int n_cand, rocm_array active_list, rocm_array cand_info, rocm_array cand_n,
                          rocm_array cand_m, rocm_array owner, rocm_array exec, rocm_array active_next,
                          uintptr_t stream_ptr) {
    constexpr const char* kFn = "delaunay_flip_decide";
    require_buf(active_list, kInt, 32, (static_cast<int64_t>(n_cand) + 3) / 4, kFn, "active_list");
    require_buf(cand_info, kInt, 32, n_cand, kFn, "cand_info");
    require_buf(cand_n, kInt, 32, n_cand, kFn, "cand_n");
    require_buf(cand_m, kInt, 32, n_cand, kFn, "cand_m");
    require_buf(owner, kInt, 32, 0, kFn, "owner");
    require_buf(exec, kInt, 32, n_cand, kFn, "exec");
    require_buf(active_next, kUInt, 8, 0, kFn, "active_next");
    launch_delaunay_flip_decide(n_cand, active_list.data(), cand_info.data(), cand_n.data(), cand_m.data(),
                                owner.data(), exec.data(), active_next.data(),
                                reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void delaunay_flip_execute(int n_cand, rocm_array active_list, rocm_array exec, rocm_array cand_n,
                           rocm_array cand_m, rocm_array slot_rank, rocm_array run_rank, int base_new,
                           rocm_array points, rocm_array tet_v, rocm_array tet_opp, rocm_array alive,
                           rocm_array flipped, rocm_array redirect, rocm_array pend, rocm_array new_tets,
                           rocm_array active_next, uintptr_t stream_ptr) {
    constexpr const char* kFn = "delaunay_flip_execute";
    require_buf(active_list, kInt, 32, (static_cast<int64_t>(n_cand) + 3) / 4, kFn, "active_list");
    require_buf(exec, kInt, 32, n_cand, kFn, "exec");
    require_buf(cand_n, kInt, 32, n_cand, kFn, "cand_n");
    require_buf(cand_m, kInt, 32, n_cand, kFn, "cand_m");
    require_buf(slot_rank, kInt, 32, n_cand, kFn, "slot_rank");
    require_buf(run_rank, kInt, 32, n_cand, kFn, "run_rank");
    require_buf(points, kFloat, 64, 0, kFn, "points");
    require_buf(tet_v, kInt, 32, extent(4, base_new), kFn, "tet_v");
    require_buf(tet_opp, kInt, 32, extent(4, base_new), kFn, "tet_opp");
    require_buf(alive, kUInt, 8, base_new, kFn, "alive");
    require_buf(flipped, kInt, 32, base_new, kFn, "flipped");
    require_buf(redirect, kInt, 32, extent(4, base_new), kFn, "redirect");
    require_buf(pend, kUInt, 8, extent(4, base_new), kFn, "pend");
    require_buf(new_tets, kInt, 32, 0, kFn, "new_tets");
    require_buf(active_next, kUInt, 8, base_new, kFn, "active_next");
    launch_delaunay_flip_execute(n_cand, active_list.data(), exec.data(), cand_n.data(), cand_m.data(),
                                 slot_rank.data(), run_rank.data(), base_new, points.data(), tet_v.data(),
                                 tet_opp.data(), alive.data(), flipped.data(), redirect.data(), pend.data(),
                                 new_tets.data(), active_next.data(), reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void delaunay_flip_fixup(int n_run, rocm_array new_tets, rocm_array flipped, rocm_array redirect, rocm_array tet_opp,
                         rocm_array pend, uintptr_t stream_ptr) {
    constexpr const char* kFn = "delaunay_flip_fixup";
    require_buf(new_tets, kInt, 32, extent(3, n_run), kFn, "new_tets");
    require_buf(flipped, kInt, 32, 0, kFn, "flipped");
    require_buf(redirect, kInt, 32, 0, kFn, "redirect");
    require_buf(tet_opp, kInt, 32, 0, kFn, "tet_opp");
    require_buf(pend, kUInt, 8, 0, kFn, "pend");
    launch_delaunay_flip_fixup(n_run, new_tets.data(), flipped.data(), redirect.data(), tet_opp.data(), pend.data(),
                               reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void delaunay_relocate_flip(int n_pts, rocm_array flipped, rocm_array new_tets, rocm_array tet_v, rocm_array points,
                            rocm_array pt_tet, uintptr_t stream_ptr) {
    constexpr const char* kFn = "delaunay_relocate_flip";
    require_buf(flipped, kInt, 32, 0, kFn, "flipped");
    require_buf(new_tets, kInt, 32, 0, kFn, "new_tets");
    require_buf(tet_v, kInt, 32, 0, kFn, "tet_v");
    require_buf(points, kFloat, 64, extent(3, n_pts), kFn, "points");
    require_buf(pt_tet, kInt, 32, n_pts, kFn, "pt_tet");
    launch_delaunay_relocate_flip(n_pts, flipped.data(), new_tets.data(), tet_v.data(), points.data(), pt_tet.data(),
                                  reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void delaunay_compact(int n_tets, rocm_array alive, rocm_array new_index, rocm_array tet_v, rocm_array tet_opp,
                      rocm_array out_tets, rocm_array out_nbr, uintptr_t stream_ptr) {
    constexpr const char* kFn = "delaunay_compact";
    require_buf(alive, kUInt, 8, n_tets, kFn, "alive");
    require_buf(new_index, kInt, 32, n_tets, kFn, "new_index");
    require_buf(tet_v, kInt, 32, extent(4, n_tets), kFn, "tet_v");
    require_buf(tet_opp, kInt, 32, extent(4, n_tets), kFn, "tet_opp");
    require_buf(out_tets, kInt, 32, 0, kFn, "out_tets");
    require_buf(out_nbr, kInt, 32, 0, kFn, "out_nbr");
    launch_delaunay_compact(n_tets, alive.data(), new_index.data(), tet_v.data(), tet_opp.data(), out_tets.data(),
                            out_nbr.data(), reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void mesh_bvh_build(rocm_array codes, rocm_array tris, int n_leaves, rocm_array child, rocm_array parent,
                    rocm_array visits, rocm_array box, uintptr_t stream_ptr) {
    constexpr const char* kFn = "mesh_bvh_build";
    require_positive(n_leaves, kFn, "n_leaves");
    const int64_t nodes = 2 * static_cast<int64_t>(n_leaves) - 1;
    require_buf(codes, kInt, 64, n_leaves, kFn, "codes");
    require_buf(tris, kFloat, 32, extent(9, n_leaves), kFn, "tris");
    require_buf(child, kInt, 32, extent(2, n_leaves - 1), kFn, "child");
    require_buf(parent, kInt, 32, nodes, kFn, "parent");
    require_buf(visits, kInt, 32, n_leaves - 1, kFn, "visits");
    require_buf(box, kFloat, 32, extent(6, nodes), kFn, "box");
    launch_mesh_bvh_build(codes.data(), tris.data(), n_leaves, child.data(), parent.data(), visits.data(), box.data(),
                          reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void closest_point_on_mesh(rocm_array points, int n_points, rocm_array tris, rocm_array tri_index, int n_leaves,
                           rocm_array child, rocm_array box, float max_dist, rocm_array dist, rocm_array closest,
                           rocm_array face, uintptr_t stream_ptr) {
    constexpr const char* kFn = "closest_point_on_mesh";
    require_positive(n_leaves, kFn, "n_leaves");
    require_buf(points, kFloat, 32, extent(3, n_points), kFn, "points");
    require_buf(tris, kFloat, 32, extent(9, n_leaves), kFn, "tris");
    require_buf(tri_index, kInt, 32, n_leaves, kFn, "tri_index");
    require_buf(child, kInt, 32, extent(2, n_leaves - 1), kFn, "child");
    require_buf(box, kFloat, 32, extent(6, 2 * static_cast<int64_t>(n_leaves) - 1), kFn, "box");
    require_buf(dist, kFloat, 32, n_points, kFn, "dist");
    require_buf(closest, kFloat, 32, extent(3, n_points), kFn, "closest");
    require_buf(face, kInt, 64, n_points, kFn, "face");
    launch_closest_point_on_mesh(points.data(), n_points, tris.data(), tri_index.data(), n_leaves, child.data(),
                                 box.data(), max_dist, dist.data(), closest.data(), face.data(),
                                 reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void closest_hit_on_mesh(rocm_array origins, rocm_array directions, int n_rays, rocm_array tris, rocm_array tri_index,
                         int n_leaves, rocm_array child, rocm_array box, float tmin, float tmax, bool cull_backface,
                         rocm_array t, rocm_array face, uintptr_t stream_ptr) {
    constexpr const char* kFn = "closest_hit_on_mesh";
    require_positive(n_leaves, kFn, "n_leaves");
    require_buf(origins, kFloat, 32, extent(3, n_rays), kFn, "origins");
    require_buf(directions, kFloat, 32, extent(3, n_rays), kFn, "directions");
    require_buf(tris, kFloat, 32, extent(9, n_leaves), kFn, "tris");
    require_buf(tri_index, kInt, 32, n_leaves, kFn, "tri_index");
    require_buf(child, kInt, 32, extent(2, n_leaves - 1), kFn, "child");
    require_buf(box, kFloat, 32, extent(6, 2 * static_cast<int64_t>(n_leaves) - 1), kFn, "box");
    require_buf(t, kFloat, 32, n_rays, kFn, "t");
    require_buf(face, kInt, 64, n_rays, kFn, "face");
    launch_closest_hit_on_mesh(origins.data(), directions.data(), n_rays, tris.data(), tri_index.data(), n_leaves,
                               child.data(), box.data(), tmin, tmax, cull_backface, t.data(), face.data(),
                               reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void connected_components(rocm_array edges, int64_t n_edges, bool edges_int64, int n_nodes, rocm_array parent,
                          rocm_array labels, uintptr_t stream_ptr) {
    constexpr const char* kFn = "connected_components";
    require_buf(edges, kInt, edges_int64 ? 64 : 32, extent(2, n_edges), kFn, "edges");
    require_buf(parent, kInt, 32, n_nodes, kFn, "parent");
    require_buf(labels, kInt, 64, n_nodes, kFn, "labels");
    launch_connected_components(edges.data(), n_edges, edges_int64, n_nodes, parent.data(), labels.data(),
                                reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void edge_collapse_checks(rocm_array verts, rocm_array faces, rocm_array offsets, rocm_array cursor, rocm_array fan,
                          rocm_array edges, rocm_array positions, int n_corners, int n_edges, float cos_threshold,
                          rocm_array flips, rocm_array skinny, rocm_array link_ok, uintptr_t stream_ptr) {
    constexpr const char* kFn = "edge_collapse_checks";
    const int64_t n_verts = static_cast<int64_t>(verts.size() / 3);
    require_buf(verts, kFloat, 32, 0, kFn, "verts");
    require_buf(faces, kInt, 32, n_corners, kFn, "faces");
    require_buf(offsets, kInt, 32, n_verts + 1, kFn, "offsets");
    require_buf(cursor, kInt, 32, n_verts, kFn, "cursor");
    require_buf(fan, kInt, 32, n_corners, kFn, "fan");
    require_buf(edges, kInt, 32, extent(2, n_edges), kFn, "edges");
    require_buf(positions, kFloat, 32, extent(3, n_edges), kFn, "positions");
    require_buf(flips, kInt, 32, n_edges, kFn, "flips");
    require_buf(skinny, kFloat, 32, n_edges, kFn, "skinny");
    require_buf(link_ok, kBool, 8, n_edges, kFn, "link_ok");
    launch_edge_collapse_checks(verts.data(), faces.data(), offsets.data(), cursor.data(), fan.data(), edges.data(),
                                positions.data(), n_corners, n_edges, cos_threshold, flips.data(), skinny.data(),
                                link_ok.data(), reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void min_cut_init(int n, int K, rocm_array nbr, rocm_array cap, rocm_array r, rocm_array rev, uintptr_t stream_ptr) {
    constexpr const char* kFn = "min_cut_init";
    require_buf(nbr, kInt, 32, extent(n, K), kFn, "nbr");
    require_buf(cap, kFloat, 32, extent(n, K), kFn, "cap");
    require_buf(r, kFloat, 32, extent(n, K), kFn, "r");
    require_buf(rev, kInt, 8, extent(n, K), kFn, "rev");
    launch_min_cut_init(n, K, nbr.data(), cap.data(), r.data(), rev.data(), reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void min_cut_global_relabel(int n, int K, rocm_array nbr, rocm_array rev, rocm_array r, rocm_array rt, float tol, int big,
                            rocm_array h, rocm_array lists, rocm_array counts, uintptr_t stream_ptr) {
    constexpr const char* kFn = "min_cut_global_relabel";
    require_buf(nbr, kInt, 32, extent(n, K), kFn, "nbr");
    require_buf(rev, kInt, 8, extent(n, K), kFn, "rev");
    require_buf(r, kFloat, 32, extent(n, K), kFn, "r");
    require_buf(rt, kFloat, 32, n, kFn, "rt");
    require_buf(h, kInt, 32, n, kFn, "h");
    require_buf(lists, kInt, 32, extent(2, n), kFn, "lists");
    require_buf(counts, kInt, 32, 2, kFn, "counts");
    launch_min_cut_global_relabel(n, K, nbr.data(), rev.data(), r.data(), rt.data(), tol, big, h.data(), lists.data(),
                                  counts.data(), reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void min_cut_rebuild_active(int n, rocm_array e, rocm_array h, float tol, int big, rocm_array list, rocm_array count,
                            rocm_array queued, uintptr_t stream_ptr) {
    constexpr const char* kFn = "min_cut_rebuild_active";
    require_buf(e, kFloat, 32, n, kFn, "e");
    require_buf(h, kInt, 32, n, kFn, "h");
    require_buf(list, kInt, 32, n, kFn, "list");
    require_buf(count, kInt, 32, 1, kFn, "count");
    require_buf(queued, kInt, 32, n, kFn, "queued");
    launch_min_cut_rebuild_active(n, e.data(), h.data(), tol, big, list.data(), count.data(), queued.data(),
                                  reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void min_cut_iterate(int n, int K, rocm_array nbr, rocm_array rev, rocm_array r, rocm_array e, rocm_array rt, rocm_array h,
                     rocm_array inc, rocm_array hn, float tol, int big, rocm_array list, rocm_array count,
                     rocm_array list_out, rocm_array count_out, rocm_array queued, rocm_array active, uintptr_t stream_ptr) {
    constexpr const char* kFn = "min_cut_iterate";
    require_buf(nbr, kInt, 32, extent(n, K), kFn, "nbr");
    require_buf(rev, kInt, 8, extent(n, K), kFn, "rev");
    require_buf(r, kFloat, 32, extent(n, K), kFn, "r");
    require_buf(e, kFloat, 32, n, kFn, "e");
    require_buf(rt, kFloat, 32, n, kFn, "rt");
    require_buf(h, kInt, 32, n, kFn, "h");
    require_buf(inc, kFloat, 32, extent(n, K), kFn, "inc");
    require_buf(hn, kInt, 32, n, kFn, "hn");
    require_buf(list, kInt, 32, n, kFn, "list");
    require_buf(count, kInt, 32, 1, kFn, "count");
    require_buf(list_out, kInt, 32, n, kFn, "list_out");
    require_buf(count_out, kInt, 32, 1, kFn, "count_out");
    require_buf(queued, kInt, 32, n, kFn, "queued");
    require_buf(active, kInt, 32, 1, kFn, "active");
    launch_min_cut_iterate(n, K, nbr.data(), rev.data(), r.data(), e.data(), rt.data(), h.data(), inc.data(), hn.data(),
                           tol, big, list.data(), count.data(), list_out.data(), count_out.data(), queued.data(),
                           active.data(), reinterpret_cast<hipStream_t>(stream_ptr));
    check_hip_launch();
}

void register_geometry(nb::module_& m) {
    m.def("delaunay_vote", &delaunay_vote, "Delaunay: each tet with uninserted points picks one",
          nb::arg("pt_tet"), nb::arg("pt_prio"), nb::arg("n_pts"), nb::arg("vote"), nb::arg("stream_ptr"));
    m.def("delaunay_split", &delaunay_split, "Delaunay: 1-4 split of the voting tets",
          nb::arg("n_tets"), nb::arg("vote"), nb::arg("split_rank"), nb::arg("prio_to_pt"), nb::arg("tet_v"),
          nb::arg("tet_opp"), nb::arg("alive"), nb::arg("pt_tet"), nb::arg("active"), nb::arg("stream_ptr"));
    m.def("delaunay_relocate_split", &delaunay_relocate_split, "Delaunay: move points of split tets to their child",
          nb::arg("n_pts"), nb::arg("vote"), nb::arg("split_rank"), nb::arg("base_new"), nb::arg("tet_v"),
          nb::arg("points"), nb::arg("pt_tet"), nb::arg("stream_ptr"));
    m.def("delaunay_flip_detect", &delaunay_flip_detect, "Delaunay: find 2-3 / 3-2 flips on changed tets",
          nb::arg("n_active"), nb::arg("active_list"), nb::arg("active_flag"), nb::arg("tet_v"), nb::arg("tet_opp"),
          nb::arg("points"), nb::arg("cand_info"), nb::arg("cand_n"), nb::arg("cand_m"), nb::arg("owner"),
          nb::arg("retry"), nb::arg("stream_ptr"));
    m.def("delaunay_flip_decide", &delaunay_flip_decide, "Delaunay: keep flips that own all their tets",
          nb::arg("n_cand"), nb::arg("active_list"), nb::arg("cand_info"), nb::arg("cand_n"), nb::arg("cand_m"),
          nb::arg("owner"), nb::arg("exec"), nb::arg("active_next"), nb::arg("stream_ptr"));
    m.def("delaunay_flip_execute", &delaunay_flip_execute, "Delaunay: perform the chosen flips",
          nb::arg("n_cand"), nb::arg("active_list"), nb::arg("exec"), nb::arg("cand_n"), nb::arg("cand_m"),
          nb::arg("slot_rank"), nb::arg("run_rank"), nb::arg("base_new"), nb::arg("points"), nb::arg("tet_v"),
          nb::arg("tet_opp"), nb::arg("alive"), nb::arg("flipped"), nb::arg("redirect"), nb::arg("pend"),
          nb::arg("new_tets"), nb::arg("active_next"), nb::arg("stream_ptr"));
    m.def("delaunay_flip_fixup", &delaunay_flip_fixup, "Delaunay: reconnect faces across flips of one pass",
          nb::arg("n_run"), nb::arg("new_tets"), nb::arg("flipped"), nb::arg("redirect"), nb::arg("tet_opp"),
          nb::arg("pend"), nb::arg("stream_ptr"));
    m.def("delaunay_relocate_flip", &delaunay_relocate_flip, "Delaunay: move points of flipped tets",
          nb::arg("n_pts"), nb::arg("flipped"), nb::arg("new_tets"), nb::arg("tet_v"), nb::arg("points"),
          nb::arg("pt_tet"), nb::arg("stream_ptr"));
    m.def("delaunay_compact", &delaunay_compact, "Delaunay: drop dead tets and remap neighbours",
          nb::arg("n_tets"), nb::arg("alive"), nb::arg("new_index"), nb::arg("tet_v"), nb::arg("tet_opp"),
          nb::arg("out_tets"), nb::arg("out_nbr"), nb::arg("stream_ptr"));
    m.def("mesh_bvh_build", &mesh_bvh_build, "Linear BVH over Morton-sorted triangles",
          nb::arg("codes"), nb::arg("tris"), nb::arg("n_leaves"), nb::arg("child"), nb::arg("parent"),
          nb::arg("visits"), nb::arg("box"), nb::arg("stream_ptr"));
    m.def("closest_point_on_mesh", &closest_point_on_mesh, "Closest point on a BVH'd triangle mesh per query",
          nb::arg("points"), nb::arg("n_points"), nb::arg("tris"), nb::arg("tri_index"), nb::arg("n_leaves"),
          nb::arg("child"), nb::arg("box"), nb::arg("max_dist"), nb::arg("dist"), nb::arg("closest"), nb::arg("face"),
          nb::arg("stream_ptr"));
    m.def("closest_hit_on_mesh", &closest_hit_on_mesh, "Nearest ray hit on a BVH'd triangle mesh per ray",
          nb::arg("origins"), nb::arg("directions"), nb::arg("n_rays"), nb::arg("tris"), nb::arg("tri_index"),
          nb::arg("n_leaves"), nb::arg("child"), nb::arg("box"), nb::arg("tmin"), nb::arg("tmax"),
          nb::arg("cull_backface"), nb::arg("t"), nb::arg("face"), nb::arg("stream_ptr"));
    m.def("connected_components", &connected_components, "Union-find connected components of an edge list",
          nb::arg("edges"), nb::arg("n_edges"), nb::arg("edges_int64"), nb::arg("n_nodes"), nb::arg("parent"),
          nb::arg("labels"), nb::arg("stream_ptr"));
    m.def("edge_collapse_checks", &edge_collapse_checks, "Flip / skinny / link checks for candidate edge collapses",
          nb::arg("verts"), nb::arg("faces"), nb::arg("offsets"), nb::arg("cursor"), nb::arg("fan"), nb::arg("edges"),
          nb::arg("positions"), nb::arg("n_corners"), nb::arg("n_edges"), nb::arg("cos_threshold"), nb::arg("flips"),
          nb::arg("skinny"), nb::arg("link_ok"), nb::arg("stream_ptr"));
    m.def("min_cut_init", &min_cut_init, "Residuals and reverse slots for min_cut", nb::arg("n"), nb::arg("K"),
          nb::arg("nbr"), nb::arg("cap"), nb::arg("r"), nb::arg("rev"), nb::arg("stream_ptr"));
    m.def("min_cut_global_relabel", &min_cut_global_relabel, "Heights by BFS from the sink", nb::arg("n"), nb::arg("K"),
          nb::arg("nbr"), nb::arg("rev"), nb::arg("r"), nb::arg("rt"), nb::arg("tol"), nb::arg("big"), nb::arg("h"),
          nb::arg("lists"), nb::arg("counts"), nb::arg("stream_ptr"));
    m.def("min_cut_rebuild_active", &min_cut_rebuild_active, "Active list from scratch", nb::arg("n"), nb::arg("e"),
          nb::arg("h"), nb::arg("tol"), nb::arg("big"), nb::arg("list"), nb::arg("count"), nb::arg("queued"),
          nb::arg("stream_ptr"));
    m.def("min_cut_iterate", &min_cut_iterate, "One push-relabel iteration over the active list", nb::arg("n"),
          nb::arg("K"), nb::arg("nbr"), nb::arg("rev"), nb::arg("r"), nb::arg("e"), nb::arg("rt"), nb::arg("h"),
          nb::arg("inc"), nb::arg("hn"), nb::arg("tol"), nb::arg("big"), nb::arg("list"), nb::arg("count"),
          nb::arg("list_out"), nb::arg("count_out"), nb::arg("queued"), nb::arg("active"), nb::arg("stream_ptr"));
}
