// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Tokha233
#pragma once
__device__ __forceinline__ unsigned tma_smem_addr(const void *p) {
  return (unsigned)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void tma_bar_init(uint64_t *b) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(tma_smem_addr(b))
               : "memory");
}
__device__ __forceinline__ void tma_bar_wait(uint64_t *b, unsigned phase) {
  unsigned done = 0;
  do {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, "
                 "[%1], %2; selp.u32 %0, 1, 0, p; }"
                 : "=r"(done)
                 : "r"(tma_smem_addr(b)), "r"(phase)
                 : "memory");
  } while (!done);
}
__device__ __forceinline__ void tma_bar_init_count(uint64_t *b,
                                                   unsigned count) {
  asm volatile(
      "mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(tma_smem_addr(b)),
      "r"(count)
      : "memory");
}
__device__ __forceinline__ void tma_bar_arrive(uint64_t *b) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(
                   tma_smem_addr(b))
               : "memory");
}
__device__ __forceinline__ void tma_issue(const CUtensorMap *map, void *dst,
                                          uint64_t *bar, int x, int y, int z) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 16384;" ::"r"(
          tma_smem_addr(bar))
      : "memory");
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%2, %3, %4}], [%5];" ::"r"(tma_smem_addr(dst)),
      "l"((uint64_t)map), "r"(x), "r"(y), "r"(z), "r"(tma_smem_addr(bar))
      : "memory");
}
