  if constexpr (HasFp8) {
    // Phase: NVFP4
#if MPA_LOW4_NVFP4
    static_assert((kCtaQ == 64 || kCtaQ == 128) && HeadDim == 128);
    if (nv_iterations != 0) {
      constexpr uint32_t threads = num_warps * 32;
      constexpr uint32_t k_data_bytes = kCtaK * (HeadDim / 2);
      constexpr uint32_t v_data_bytes = HeadDim * (kCtaK / 2);
      constexpr uint32_t k_scale_bytes = kCtaK * (HeadDim / 16);
      constexpr uint32_t v_scale_bytes = HeadDim * (kCtaK / 16);
      const uint32_t cta_thread = warp_id * 32 + lane_id;
      uint8_t* k_tile = reinterpret_cast<uint8_t*>(smem);
      uint8_t* v_tile = k_tile + k_data_bytes;
      uint8_t* k_scale_tile = v_tile + v_data_bytes;
      uint8_t* v_scale_tile = k_scale_tile + k_scale_bytes;

      const uint8_t* q_base = reinterpret_cast<const uint8_t*>(
#if MPA_THREE_PHASE
          q4
#else
          q8
#endif
          ) +
          ((batch_id * num_qo_heads + head_id) * qo_len +
           query_block * kCtaQ) * (HeadDim / 2);
      const uint8_t* q_scale_base =
#if MPA_THREE_PHASE
          q4_scale
#else
          q_scale
#endif
          +
          ((batch_id * num_qo_heads + head_id) * qo_len +
           query_block * kCtaQ) * (HeadDim / 16);
      uint32_t q_data[num_tiles_q][HeadDim / 64][4];
      uint32_t q_scale_data[num_tiles_q][HeadDim / 64];
      load_nvfp4_query<HeadDim>(
          q_base, q_scale_base, q_data, q_scale_data);

      int32_t* low_lut = fp8_lut + metadata_row * num_physical_stages;
      const float nv_score_scale = base2_softmax_scale *
          q_global_scale[0] * k_global_scale[0];
      const uint32_t nv_v_stride =
#if MPA_THREE_PHASE
          kv_len;
#else
          padded_kv_len;
#endif
      for (uint32_t iteration = 0; iteration < nv_iterations; ++iteration) {
        const uint32_t absolute_stage =
            static_cast<uint32_t>(low_lut[iteration]);
        const uint8_t* k_source = reinterpret_cast<const uint8_t*>(
#if MPA_THREE_PHASE
            k4
#else
            k8
#endif
            ) +
            ((batch_id * num_kv_heads + kv_head) * kv_len +
             absolute_stage * kCtaK) * (HeadDim / 2);
        const uint8_t* k_scale_source =
#if MPA_THREE_PHASE
            k4_scale
#else
            k_scale
#endif
            +
            ((batch_id * num_kv_heads + kv_head) * kv_len +
             absolute_stage * kCtaK) * (HeadDim / 16);
        copy_nvfp4_async_swizzled<kCtaK, HeadDim / 2, threads>(
            k_tile, k_source, cta_thread, HeadDim / 2);
        copy_nvfp4_async_contiguous<k_scale_bytes, threads>(
            k_scale_tile, k_scale_source, cta_thread);
        cp_async::commit_group();

        const uint8_t* v_source =
            reinterpret_cast<const uint8_t*>(
#if MPA_THREE_PHASE
                v4
#else
                v8
#endif
            ) +
            ((batch_id * num_kv_heads + kv_head) * HeadDim) *
                (nv_v_stride / 2) +
            absolute_stage * (kCtaK / 2);
        const uint8_t* v_scale_source =
#if MPA_THREE_PHASE
            v4_scale
#else
            v_scale
#endif
            +
            ((batch_id * num_kv_heads + kv_head) * num_physical_stages +
             absolute_stage) * v_scale_bytes;
        copy_nvfp4_async_swizzled<HeadDim, kCtaK / 2, threads>(
            v_tile, v_source, cta_thread, nv_v_stride / 2);
        copy_nvfp4_async_contiguous<v_scale_bytes, threads>(
            v_scale_tile, v_scale_source, cta_thread);
        cp_async::commit_group();

        cp_async::wait_group<1>();
        __syncthreads();
        float rs[num_tiles_q][num_tiles_k][8];
        compute_nvfp4_qk<HeadDim>(
            q_data, q_scale_data, k_tile, k_scale_tile, rs);
        const uint32_t valid_k = static_cast<uint32_t>(
            valid_k_counts[batch_id * num_physical_stages + absolute_stage]);
        const uint32_t k_index_base =
            get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
            2 * (lane_id % 4);
        apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(
            k_index_base, rs, valid_k);
        update_mdo<
            num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(
            rs, ro, m, d, nv_score_scale);
        accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(rs, d);
        const NvProbabilityFragment probability0 =
            prepare_nvfp4_probability(rs[0]);
        NvProbabilityFragment probability1{};
        if constexpr (num_tiles_q == 2) {
          probability1 = prepare_nvfp4_probability(rs[1]);
        }
        cp_async::wait_group<0>();
        __syncthreads();
        accumulate_nvfp4_pv<HeadDim>(
            ro[0], probability0, v_tile, v_scale_tile);
        if constexpr (num_tiles_q == 2) {
          accumulate_nvfp4_pv<HeadDim>(
              ro[1], probability1, v_tile, v_scale_tile);
        }
        __syncthreads();
      }

      const float output_scale =
          v_global_scale[0] * kNvProbabilityGlobalScale;
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
#pragma unroll
        for (uint32_t fv = 0; fv < num_tiles_v; ++fv) {
#pragma unroll
          for (uint32_t element = 0; element < 8; ++element) {
            ro[fq][fv][element] *= output_scale;
          }
        }
      }
    }
#endif

    // Phase: MXFP8
#if (!MPA_LOW4_NVFP4 && !MPA_MIDDLE_INT8) || MPA_MIDDLE_MXFP8
    if (low_iterations != 0) {
      using QKSmem = LowQKSmem<HeadDim>;
      QKSmem smem_q8(smem);
      QKSmem smem_k8(smem + kCtaQ * HeadDim);
      LowVSmem smem_v8(smem + (kCtaQ + kCtaK) * HeadDim);
      constexpr uint32_t kLowScaleSmemBytes =
          kCtaK * (HeadDim / 32) + HeadDim * 2;
      uint8_t* k_scale_tile = reinterpret_cast<uint8_t*>(
          smem + (kCtaQ + 2 * kCtaK) * HeadDim);
      uint8_t* v_scale_tile =
          k_scale_tile + kLowScaleSmemBytes - HeadDim * 2;
      const uint32_t cta_thread = warp_id * 32 + lane_id;

      constexpr uint32_t qk_line_lanes = HeadDim == 64 ? 4 : 8;
      constexpr uint32_t qk_lines_per_warp = 32 / qk_line_lanes;
      constexpr uint32_t qk_smem_iters_row =
          HeadDim / (qk_line_lanes * kPackInt8);
      constexpr uint32_t q_smem_iters_col =
          kCtaQ / (num_warps * qk_lines_per_warp);
      constexpr uint32_t k_smem_iters_col =
          kCtaK / (num_warps * qk_lines_per_warp);
      constexpr uint32_t v_line_lanes = 4;
      constexpr uint32_t v_lines_per_warp = 8;
      constexpr uint32_t v_smem_iters_row =
          kCtaK / (v_line_lanes * kPackFp8);
      constexpr uint32_t v_smem_iters_col =
          HeadDim / (num_warps * v_lines_per_warp);

      int8_t* q_lane =
          q8 + ((batch_id * num_qo_heads + head_id) * qo_len +
                query_block * kCtaQ +
                kCtaQ / num_warps * warp_id + lane_id / qk_line_lanes) *
                   HeadDim +
          (lane_id % qk_line_lanes) * kPackInt8;
      uint32_t q_smem_load = smem_q8.get_permuted_offset(
          warp_id * qk_lines_per_warp * q_smem_iters_col +
              lane_id / qk_line_lanes,
          lane_id % qk_line_lanes);
      const uint32_t q_load_row =
          query_block * kCtaQ + kCtaQ / num_warps * warp_id +
          lane_id / qk_line_lanes;
      load_global_to_share<
          qk_line_lanes, qk_lines_per_warp, qk_smem_iters_row,
          q_smem_iters_col, (HeadDim == 64 ? SwizzleMode::k64B
                                           : SwizzleMode::k128B),
          HeadDim / kPackInt8, kCtaQ>(
          q_lane, q_smem_load, HeadDim, smem_q8, q_load_row, qo_len);
      cp_async::commit_group();
      cp_async::wait_group<0>();
      __syncthreads();

      const uint8_t* q_mxfp8_scale = q_scale +
          ((batch_id * num_qo_heads + head_id) * qo_len +
           query_block * kCtaQ) * (HeadDim / 32);
      int32_t* low_lut = fp8_lut + metadata_row * num_physical_stages +
          nv_iterations;
      uint32_t absolute_stage = 0;

      // Shared by pure and mixed MXFP8 specializations; all async groups are
      // drained before the optional FP16 phase starts.
      if constexpr (HasFp8) {
#if MPA_DENSE_SEQUENTIAL
        absolute_stage = 0;
#else
        absolute_stage = static_cast<uint32_t>(low_lut[0]);
#endif
        int8_t* k_lane =
            k8 + ((batch_id * num_kv_heads + kv_head) * kv_len +
                  absolute_stage * kCtaK +
                  kCtaK / num_warps * warp_id +
                  lane_id / qk_line_lanes) *
                     HeadDim +
            (lane_id % qk_line_lanes) * kPackInt8;
        uint32_t k_smem_load = smem_k8.get_permuted_offset(
            warp_id * qk_lines_per_warp * k_smem_iters_col +
                lane_id / qk_line_lanes,
            lane_id % qk_line_lanes);
        uint32_t k_load_row =
            absolute_stage * kCtaK + kCtaK / num_warps * warp_id +
            lane_id / qk_line_lanes;
        load_global_to_share<
            qk_line_lanes, qk_lines_per_warp, qk_smem_iters_row,
            k_smem_iters_col, (HeadDim == 64 ? SwizzleMode::k64B
                                             : SwizzleMode::k128B),
            HeadDim / kPackInt8, kCtaK>(
            k_lane, k_smem_load, HeadDim, smem_k8, k_load_row, kv_len);
        if (cta_thread < kCtaK * (HeadDim / 32) / 16) {
          const uint8_t* k_scale_stage = k_scale +
              ((batch_id * num_kv_heads + kv_head) * kv_len +
               absolute_stage * kCtaK) * (HeadDim / 32);
          cp_async::load_128b<cp_async::PrefetchMode::kNoPrefetch>(
              k_scale_tile + cta_thread * 16,
              k_scale_stage + cta_thread * 16);
        }
        cp_async::commit_group();

        int8_t* v_lane =
            reinterpret_cast<int8_t*>(v8) +
            ((batch_id * num_kv_heads + kv_head) * HeadDim +
             HeadDim / num_warps * warp_id + lane_id / v_line_lanes) *
                padded_kv_len +
            absolute_stage * kCtaK +
            (lane_id % v_line_lanes) * kPackFp8;
        uint32_t v_smem_load = smem_v8.get_permuted_offset(
            warp_id * v_lines_per_warp * v_smem_iters_col +
                lane_id / v_line_lanes,
            lane_id % v_line_lanes);
        load_fp8_V_global_to_share<
            v_line_lanes, v_lines_per_warp, v_smem_iters_row,
            v_smem_iters_col, SwizzleMode::k64B, kCtaK / kPackFp8,
            kCtaK>(v_lane, v_smem_load, padded_kv_len, smem_v8);
        if (cta_thread < HeadDim * 2 / 16) {
          const uint8_t* v_scale_stage = v_scale +
              ((batch_id * num_kv_heads + kv_head) *
                   num_physical_stages +
               absolute_stage) * (HeadDim * 2);
          cp_async::load_128b<cp_async::PrefetchMode::kNoPrefetch>(
              v_scale_tile + cta_thread * 16,
              v_scale_stage + cta_thread * 16);
        }
        cp_async::commit_group();

        for (uint32_t iteration = 0;
             iteration < low_iterations;
             ++iteration) {
          cp_async::wait_group<1>();
          __syncthreads();

          float rs[num_tiles_q][num_tiles_k][8];
          compute_mxfp8_qk<HeadDim>(
              smem_q8, smem_k8, q_mxfp8_scale, k_scale_tile, rs);

          uint32_t next_stage = 0;
          const bool has_next = iteration + 1 < low_iterations;
          if (has_next) {
            __syncthreads();
#if MPA_DENSE_SEQUENTIAL
            next_stage = iteration + 1;
#else
            next_stage = static_cast<uint32_t>(low_lut[iteration + 1]);
#endif
            k_lane =
                k8 + ((batch_id * num_kv_heads + kv_head) * kv_len +
                      next_stage * kCtaK +
                      kCtaK / num_warps * warp_id +
                      lane_id / qk_line_lanes) *
                         HeadDim +
                (lane_id % qk_line_lanes) * kPackInt8;
            k_load_row =
                next_stage * kCtaK + kCtaK / num_warps * warp_id +
                lane_id / qk_line_lanes;
            load_global_to_share<
                qk_line_lanes, qk_lines_per_warp, qk_smem_iters_row,
                k_smem_iters_col, (HeadDim == 64 ? SwizzleMode::k64B
                                                 : SwizzleMode::k128B),
                HeadDim / kPackInt8, kCtaK>(
                k_lane, k_smem_load, HeadDim, smem_k8, k_load_row, kv_len);
            if (cta_thread < kCtaK * (HeadDim / 32) / 16) {
              const uint8_t* k_scale_stage = k_scale +
                  ((batch_id * num_kv_heads + kv_head) * kv_len +
                   next_stage * kCtaK) * (HeadDim / 32);
              cp_async::load_128b<cp_async::PrefetchMode::kNoPrefetch>(
                  k_scale_tile + cta_thread * 16,
                  k_scale_stage + cta_thread * 16);
            }
            cp_async::commit_group();
          }

          const uint32_t valid_k = static_cast<uint32_t>(
              valid_k_counts[batch_id * num_physical_stages +
                             absolute_stage]);
          const uint32_t k_index_base =
              get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
              2 * (lane_id % 4);
          apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(
              k_index_base, rs, valid_k);
          update_mdo<
              num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(
              rs, ro, m, d, base2_softmax_scale);
          accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(
              rs, d);
          const MxProbabilityFragment probability0 =
              prepare_mxfp8_probability<0>(rs[0]);
          const MxProbabilityFragment probability1 =
              prepare_mxfp8_probability<1>(rs[0]);

          if (has_next) {
            cp_async::wait_group<1>();
          } else {
            cp_async::wait_group<0>();
          }
          __syncthreads();
          accumulate_prepared_mxfp8_pv<0, HeadDim>(
              ro[0], probability0, smem_v8, v_scale_tile);
          accumulate_prepared_mxfp8_pv<1, HeadDim>(
              ro[0], probability1, smem_v8, v_scale_tile);
#pragma unroll
          for (uint32_t fq = 1; fq < num_tiles_q; ++fq) {
            accumulate_mxfp8_pv<0, HeadDim>(
                ro[fq], rs[fq], smem_v8, v_scale_tile);
            accumulate_mxfp8_pv<1, HeadDim>(
                ro[fq], rs[fq], smem_v8, v_scale_tile);
          }

          if (has_next) {
            __syncthreads();
            absolute_stage = next_stage;
            v_lane =
                reinterpret_cast<int8_t*>(v8) +
                ((batch_id * num_kv_heads + kv_head) * HeadDim +
                 HeadDim / num_warps * warp_id + lane_id / v_line_lanes) *
                    padded_kv_len +
                absolute_stage * kCtaK +
                (lane_id % v_line_lanes) * kPackFp8;
            load_fp8_V_global_to_share<
                v_line_lanes, v_lines_per_warp, v_smem_iters_row,
                v_smem_iters_col, SwizzleMode::k64B, kCtaK / kPackFp8,
                kCtaK>(v_lane, v_smem_load, padded_kv_len, smem_v8);
            if (cta_thread < HeadDim * 2 / 16) {
              const uint8_t* v_scale_stage = v_scale +
                  ((batch_id * num_kv_heads + kv_head) *
                       num_physical_stages +
                   absolute_stage) * (HeadDim * 2);
              cp_async::load_128b<cp_async::PrefetchMode::kNoPrefetch>(
                  v_scale_tile + cta_thread * 16,
                  v_scale_stage + cta_thread * 16);
            }
            cp_async::commit_group();
          } else {
            __syncthreads();
          }
        }
      } else {
      for (uint32_t iteration = 0; iteration < low_iterations; ++iteration) {
#if defined(MPA_K64_BLOCK_MODE)
          // Native K64 routes store absolute physical-stage IDs.  The donor
          // Q128 path uses positive deltas so that its tightly pipelined
          // all-FP8 loop can update one running stage index instead.
          absolute_stage = static_cast<uint32_t>(low_lut[iteration]);
#else
          absolute_stage += static_cast<uint32_t>(low_lut[iteration]);
#endif

          int8_t* k_lane =
              k8 + ((batch_id * num_kv_heads + kv_head) * kv_len +
                    absolute_stage * kCtaK +
                    kCtaK / num_warps * warp_id +
                    lane_id / qk_line_lanes) *
                       HeadDim +
              (lane_id % qk_line_lanes) * kPackInt8;
          uint32_t k_smem_load = smem_k8.get_permuted_offset(
              warp_id * qk_lines_per_warp * k_smem_iters_col +
                  lane_id / qk_line_lanes,
              lane_id % qk_line_lanes);
          const uint32_t k_load_row =
              absolute_stage * kCtaK + kCtaK / num_warps * warp_id +
              lane_id / qk_line_lanes;
          load_global_to_share<
              qk_line_lanes, qk_lines_per_warp, qk_smem_iters_row,
              k_smem_iters_col, (HeadDim == 64 ? SwizzleMode::k64B
                                               : SwizzleMode::k128B),
              HeadDim / kPackInt8, kCtaK>(
              k_lane, k_smem_load, HeadDim, smem_k8, k_load_row, kv_len);
          if (cta_thread < kCtaK * (HeadDim / 32) / 16) {
            const uint8_t* k_scale_stage = k_scale +
                ((batch_id * num_kv_heads + kv_head) * kv_len +
                 absolute_stage * kCtaK) * (HeadDim / 32);
            cp_async::load_128b<cp_async::PrefetchMode::kNoPrefetch>(
                k_scale_tile + cta_thread * 16,
                k_scale_stage + cta_thread * 16);
          }
          cp_async::commit_group();
          cp_async::wait_group<0>();
          __syncthreads();

          float rs[num_tiles_q][num_tiles_k][8];
          compute_mxfp8_qk<HeadDim>(
              smem_q8, smem_k8, q_mxfp8_scale, k_scale_tile, rs);

          // MXFP8 V-copy/softmax overlap: V has a disjoint shared tile, so
          // start its copy after QK and hide it under masking/online softmax.
          int8_t* v_lane =
              reinterpret_cast<int8_t*>(v8) +
              ((batch_id * num_kv_heads + kv_head) * HeadDim +
               HeadDim / num_warps * warp_id + lane_id / v_line_lanes) *
                  padded_kv_len +
              absolute_stage * kCtaK +
              (lane_id % v_line_lanes) * kPackFp8;
          uint32_t v_smem_load = smem_v8.get_permuted_offset(
              warp_id * v_lines_per_warp * v_smem_iters_col +
                  lane_id / v_line_lanes,
              lane_id % v_line_lanes);
          load_fp8_V_global_to_share<
              v_line_lanes, v_lines_per_warp, v_smem_iters_row,
              v_smem_iters_col, SwizzleMode::k64B, kCtaK / kPackFp8,
              kCtaK>(v_lane, v_smem_load, padded_kv_len, smem_v8);
          if (cta_thread < HeadDim * 2 / 16) {
            const uint8_t* v_scale_stage = v_scale +
                ((batch_id * num_kv_heads + kv_head) *
                     num_physical_stages +
                 absolute_stage) * (HeadDim * 2);
            cp_async::load_128b<cp_async::PrefetchMode::kNoPrefetch>(
                v_scale_tile + cta_thread * 16,
                v_scale_stage + cta_thread * 16);
          }
          cp_async::commit_group();

          if (
#if defined(MPA_K64_BLOCK_MODE)
              true
#else
              iteration + 1 == low_iterations
#endif
          ) {
#if defined(MPA_K64_BLOCK_MODE)
            // A logical 2-D tile may occupy fewer than 64 lanes even when it
            // is not the final physical stage (8x7 is the common example).
            // Mask against this selected stage's exact valid count, not the
            // global padded K capacity.
            const uint32_t valid_k = static_cast<uint32_t>(
                valid_k_counts[batch_id * num_physical_stages +
                               absolute_stage]);
            const uint32_t k_index_base =
                get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
                2 * (lane_id % 4);
            apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(
                k_index_base, rs, valid_k);
#else
            const uint32_t k_index_base =
                absolute_stage * kCtaK +
                get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
                2 * (lane_id % 4);
            apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(
                k_index_base, rs, kv_len);
#endif
          }
          update_mdo<
              num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(
              rs, ro, m, d, base2_softmax_scale);
          accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(
              rs, d);
          cp_async::wait_group<0>();
          __syncthreads();

          accumulate_mxfp8_pv<0, HeadDim>(
              ro[0], rs[0], smem_v8, v_scale_tile);
          accumulate_mxfp8_pv<1, HeadDim>(
              ro[0], rs[0], smem_v8, v_scale_tile);
          __syncthreads();
      }
      }
    }
#endif

    // Phase: INT8
#if MPA_MIDDLE_INT8
    if (low_iterations != 0) {
      using QKSmem = LowQKSmem<HeadDim>;
      QKSmem smem_q8(smem);
      QKSmem smem_k8(smem + kCtaQ * HeadDim);
      LowVSmem smem_v8(smem + (kCtaQ + kCtaK) * HeadDim);
      constexpr uint32_t qk_line_lanes = HeadDim == 64 ? 4 : 8;
      constexpr uint32_t qk_lines_per_warp = 32 / qk_line_lanes;
      constexpr uint32_t qk_smem_iters_row =
          HeadDim / (qk_line_lanes * kPackInt8);
      constexpr uint32_t q_smem_iters_col =
          kCtaQ / (num_warps * qk_lines_per_warp);
      constexpr uint32_t k_smem_iters_col =
          kCtaK / (num_warps * qk_lines_per_warp);
      constexpr uint32_t v_line_lanes = 4;
      constexpr uint32_t v_lines_per_warp = 8;
      constexpr uint32_t v_smem_iters_row =
          kCtaK / (v_line_lanes * kPackFp8);
      constexpr uint32_t v_smem_iters_col =
          HeadDim / (num_warps * v_lines_per_warp);

      int8_t* q_lane =
          q8 + ((batch_id * num_qo_heads + head_id) * qo_len +
                query_block * kCtaQ + kCtaQ / num_warps * warp_id +
                lane_id / qk_line_lanes) * HeadDim +
          (lane_id % qk_line_lanes) * kPackInt8;
      uint32_t q_smem_load = smem_q8.get_permuted_offset(
          warp_id * qk_lines_per_warp * q_smem_iters_col +
              lane_id / qk_line_lanes,
          lane_id % qk_line_lanes);
      const uint32_t q_load_row =
          query_block * kCtaQ + kCtaQ / num_warps * warp_id +
          lane_id / qk_line_lanes;
      load_global_to_share<
          qk_line_lanes, qk_lines_per_warp, qk_smem_iters_row,
          q_smem_iters_col, SwizzleMode::k128B, HeadDim / kPackInt8,
          kCtaQ>(
          q_lane, q_smem_load, HeadDim, smem_q8, q_load_row, qo_len);
      cp_async::commit_group();
      cp_async::wait_group<0>();
      __syncthreads();

#if !MPA_THREE_PHASE
      const uint32_t q_smem_mma = smem_q8.get_permuted_offset(
          get_warp_idx_q<num_warps_q, num_warps_k>() * kWarpQ +
              lane_id % 16,
          lane_id / 16);
      const uint32_t k_smem_mma = smem_k8.get_permuted_offset(
          lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
#endif
      const float q_dequant_scale =
          q_scale[(batch_id * num_qo_heads + head_id) * num_query_blocks +
                  query_block];
      const float* v_scale_base =
          v_scale + (batch_id * num_kv_heads + kv_head) * HeadDim;
      if (nv_iterations != 0) {
        enter_fp8_probability_and_v_domains(ro, d, v_scale_base);
      }
      uint32_t absolute_stage = 0;

      // Keep the inherited Sparge/Sage copy/compute pipeline for every INT8
      // phase. K and V occupy disjoint shared-memory regions:
      // with two committed groups outstanding, wait_group<1> exposes the
      // older operand while the newer operand continues to overlap compute.
#if !MPA_DENSE_SEQUENTIAL
        int32_t* low_delta = fp8_lut +
            metadata_row * num_physical_stages + nv_iterations;
#endif
        const float* k_scale_base =
            k_scale + (batch_id * num_kv_heads + kv_head) *
                          num_physical_stages;

        // Prologue: Q is already resident.  Issue predicated K0 followed by
        // padded V0, leaving both groups outstanding.
#if MPA_DENSE_SEQUENTIAL
        absolute_stage = 0;
#elif defined(MPA_K64_BLOCK_MODE)
        absolute_stage = static_cast<uint32_t>(*low_delta++);
#else
        absolute_stage += static_cast<uint32_t>(*low_delta++);
#endif
        int8_t* k_lane =
            k8 + ((batch_id * num_kv_heads + kv_head) * kv_len +
                  absolute_stage * kCtaK +
                  kCtaK / num_warps * warp_id + lane_id / qk_line_lanes) *
                     HeadDim +
            (lane_id % qk_line_lanes) * kPackInt8;
        uint32_t k_smem_load = smem_k8.get_permuted_offset(
            warp_id * qk_lines_per_warp * k_smem_iters_col +
                lane_id / qk_line_lanes,
            lane_id % qk_line_lanes);
        uint32_t k_load_row =
            absolute_stage * kCtaK + kCtaK / num_warps * warp_id +
            lane_id / qk_line_lanes;
        load_global_to_share<
            qk_line_lanes, qk_lines_per_warp, qk_smem_iters_row,
            k_smem_iters_col, (HeadDim == 64 ? SwizzleMode::k64B
                                             : SwizzleMode::k128B),
            HeadDim / kPackInt8, kCtaK>(
            k_lane, k_smem_load, HeadDim, smem_k8, k_load_row, kv_len);
        cp_async::commit_group();

        float current_dequant_scale =
            q_dequant_scale * k_scale_base[absolute_stage];

        int8_t* v_lane =
            reinterpret_cast<int8_t*>(v8) +
            ((batch_id * num_kv_heads + kv_head) * HeadDim +
             HeadDim / num_warps * warp_id + lane_id / v_line_lanes) *
                padded_kv_len +
            absolute_stage * kCtaK +
            (lane_id % v_line_lanes) * kPackFp8;
        uint32_t v_smem_load = smem_v8.get_permuted_offset(
            warp_id * v_lines_per_warp * v_smem_iters_col +
                lane_id / v_line_lanes,
            lane_id % v_line_lanes);
        load_fp8_V_global_to_share<
            v_line_lanes, v_lines_per_warp, v_smem_iters_row,
            v_smem_iters_col, SwizzleMode::k64B, kCtaK / kPackFp8,
            kCtaK>(v_lane, v_smem_load, padded_kv_len, smem_v8);
        cp_async::commit_group();

        // Positions [0, n-3] retain donor rounding: the integer score is
        // converted to float without pre-dequantization and q*k is folded
        // into the softmax scale.
        for (uint32_t iteration = 0;
             iteration + 2 < low_iterations;
             ++iteration) {
          cp_async::wait_group<1>();
          __syncthreads();

          union LowScoreStorage {
            int32_t as_int[num_tiles_q][num_tiles_k][8];
            float as_float[num_tiles_q][num_tiles_k][8];
          } rs_storage;
#if MPA_THREE_PHASE
          uint32_t q_offset = smem_q8.get_permuted_offset(
              get_warp_idx_q<num_warps_q, num_warps_k>() * kWarpQ +
                  lane_id % 16,
              lane_id / 16);
          uint32_t k_offset = smem_k8.get_permuted_offset(
              lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
#else
          uint32_t q_offset = q_smem_mma;
          uint32_t k_offset = k_smem_mma;
#endif
          compute_int_qk<
              num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
              low_qk_inner, (HeadDim == 64 ? SwizzleMode::k64B
                                           : SwizzleMode::k128B),
              HeadDim / kPackInt8, DataType::kInt8>(
              smem_q8, smem_k8, rs_storage.as_int, q_offset, k_offset);
#pragma unroll
          for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; ++fk) {
#pragma unroll
              for (uint32_t element = 0; element < 8; ++element) {
                rs_storage.as_float[fq][fk][element] = __int2float_rz(
                    rs_storage.as_int[fq][fk][element]);
              }
            }
          }
#if defined(MPA_K64_BLOCK_MODE)
          {
            const uint32_t valid_k = static_cast<uint32_t>(
                valid_k_counts[batch_id * num_physical_stages +
                               absolute_stage]);
            const uint32_t k_index_base =
                get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
                2 * (lane_id % 4);
            apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(
                k_index_base, rs_storage.as_float, valid_k);
          }
#endif
          update_mdo<
              num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(
              rs_storage.as_float, ro, m, d,
              base2_softmax_scale * current_dequant_scale);
          accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(
              rs_storage.as_float, d);
          uint32_t rs_fp8[num_tiles_q][num_tiles_k / 2][4];
          RS_32_to_8<num_tiles_q, num_tiles_k>(
              rs_storage.as_float, rs_fp8);

          // All stages reached here precede the final active stage, hence K
          // is a full physical tile and the donor unpredicated copy is safe.
          __syncthreads();
#if MPA_DENSE_SEQUENTIAL
          absolute_stage = iteration + 1;
#elif defined(MPA_K64_BLOCK_MODE)
          absolute_stage = static_cast<uint32_t>(*low_delta++);
#else
          absolute_stage += static_cast<uint32_t>(*low_delta++);
#endif
          k_lane =
              k8 + ((batch_id * num_kv_heads + kv_head) * kv_len +
                    absolute_stage * kCtaK +
                    kCtaK / num_warps * warp_id +
                    lane_id / qk_line_lanes) *
                       HeadDim +
              (lane_id % qk_line_lanes) * kPackInt8;
          load_global_to_share<
              qk_line_lanes, qk_lines_per_warp, qk_smem_iters_row,
              k_smem_iters_col, (HeadDim == 64 ? SwizzleMode::k64B
                                               : SwizzleMode::k128B),
              HeadDim / kPackInt8, kCtaK>(
              k_lane, k_smem_load, HeadDim, smem_k8);
          cp_async::commit_group();

          cp_async::wait_group<1>();
          __syncthreads();
          compute_fp8_sv_inst_buf_fp16_accu<
              num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
              num_tiles_v, SwizzleMode::k64B, kCtaK / kPackFp8>(
              smem_v8, rs_fp8, ro, d);

          __syncthreads();
          v_lane =
              reinterpret_cast<int8_t*>(v8) +
              ((batch_id * num_kv_heads + kv_head) * HeadDim +
               HeadDim / num_warps * warp_id + lane_id / v_line_lanes) *
                  padded_kv_len +
              absolute_stage * kCtaK +
              (lane_id % v_line_lanes) * kPackFp8;
          load_fp8_V_global_to_share<
              v_line_lanes, v_lines_per_warp, v_smem_iters_row,
              v_smem_iters_col, SwizzleMode::k64B, kCtaK / kPackFp8,
              kCtaK>(v_lane, v_smem_load, padded_kv_len, smem_v8);
          cp_async::commit_group();
          current_dequant_scale =
              q_dequant_scale * k_scale_base[absolute_stage];
        }

        // The second-last active position pre-dequantizes its score before
        // softmax, matching donor instruction rounding exactly.  Its next K
        // copy is the only prefetched K that may be a physical tail.
        if (low_iterations > 1) {
          cp_async::wait_group<1>();
          __syncthreads();

          union LowScoreStorage {
            int32_t as_int[num_tiles_q][num_tiles_k][8];
            float as_float[num_tiles_q][num_tiles_k][8];
          } rs_storage;
#if MPA_THREE_PHASE
          uint32_t q_offset = smem_q8.get_permuted_offset(
              get_warp_idx_q<num_warps_q, num_warps_k>() * kWarpQ +
                  lane_id % 16,
              lane_id / 16);
          uint32_t k_offset = smem_k8.get_permuted_offset(
              lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
#else
          uint32_t q_offset = q_smem_mma;
          uint32_t k_offset = k_smem_mma;
#endif
          compute_int_qk<
              num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
              low_qk_inner, (HeadDim == 64 ? SwizzleMode::k64B
                                           : SwizzleMode::k128B),
              HeadDim / kPackInt8, DataType::kInt8>(
              smem_q8, smem_k8, rs_storage.as_int, q_offset, k_offset);
#pragma unroll
          for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; ++fk) {
#pragma unroll
              for (uint32_t element = 0; element < 8; ++element) {
                const float score = __int2float_rz(
                    rs_storage.as_int[fq][fk][element]);
                rs_storage.as_float[fq][fk][element] =
                    score * current_dequant_scale;
              }
            }
          }
#if defined(MPA_K64_BLOCK_MODE)
          {
            const uint32_t valid_k = static_cast<uint32_t>(
                valid_k_counts[batch_id * num_physical_stages +
                               absolute_stage]);
            const uint32_t k_index_base =
                get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
                2 * (lane_id % 4);
            apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(
                k_index_base, rs_storage.as_float, valid_k);
          }
#endif

          __syncthreads();
#if MPA_DENSE_SEQUENTIAL
          absolute_stage = low_iterations - 1;
#elif defined(MPA_K64_BLOCK_MODE)
          absolute_stage = static_cast<uint32_t>(*low_delta++);
#else
          absolute_stage += static_cast<uint32_t>(*low_delta++);
#endif
          k_lane =
              k8 + ((batch_id * num_kv_heads + kv_head) * kv_len +
                    absolute_stage * kCtaK +
                    kCtaK / num_warps * warp_id +
                    lane_id / qk_line_lanes) *
                       HeadDim +
              (lane_id % qk_line_lanes) * kPackInt8;
          k_load_row =
              absolute_stage * kCtaK + kCtaK / num_warps * warp_id +
              lane_id / qk_line_lanes;
          load_global_to_share<
              qk_line_lanes, qk_lines_per_warp, qk_smem_iters_row,
              k_smem_iters_col, (HeadDim == 64 ? SwizzleMode::k64B
                                               : SwizzleMode::k128B),
              HeadDim / kPackInt8, kCtaK>(
              k_lane, k_smem_load, HeadDim, smem_k8, k_load_row, kv_len);
          cp_async::commit_group();

          update_mdo<
              num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(
              rs_storage.as_float, ro, m, d, base2_softmax_scale);
          accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(
              rs_storage.as_float, d);
          uint32_t rs_fp8[num_tiles_q][num_tiles_k / 2][4];
          RS_32_to_8<num_tiles_q, num_tiles_k>(
              rs_storage.as_float, rs_fp8);

          __syncthreads();
          cp_async::wait_group<1>();
          __syncthreads();
          compute_fp8_sv_inst_buf_fp16_accu<
              num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
              num_tiles_v, SwizzleMode::k64B, kCtaK / kPackFp8>(
              smem_v8, rs_fp8, ro, d);

          __syncthreads();
          v_lane =
              reinterpret_cast<int8_t*>(v8) +
              ((batch_id * num_kv_heads + kv_head) * HeadDim +
               HeadDim / num_warps * warp_id + lane_id / v_line_lanes) *
                  padded_kv_len +
              absolute_stage * kCtaK +
              (lane_id % v_line_lanes) * kPackFp8;
          load_fp8_V_global_to_share<
              v_line_lanes, v_lines_per_warp, v_smem_iters_row,
              v_smem_iters_col, SwizzleMode::k64B, kCtaK / kPackFp8,
              kCtaK>(v_lane, v_smem_load, padded_kv_len, smem_v8);
          cp_async::commit_group();
          current_dequant_scale =
              q_dequant_scale * k_scale_base[absolute_stage];
        }

        // Last active position: drain K first, apply the only tail mask, then
        // drain V after producing the stage's online-softmax/P fragment.
        {
          cp_async::wait_group<1>();
          __syncthreads();

          union LowScoreStorage {
            int32_t as_int[num_tiles_q][num_tiles_k][8];
            float as_float[num_tiles_q][num_tiles_k][8];
          } rs_storage;
#if MPA_THREE_PHASE
          uint32_t q_offset = smem_q8.get_permuted_offset(
              get_warp_idx_q<num_warps_q, num_warps_k>() * kWarpQ +
                  lane_id % 16,
              lane_id / 16);
          uint32_t k_offset = smem_k8.get_permuted_offset(
              lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
#else
          uint32_t q_offset = q_smem_mma;
          uint32_t k_offset = k_smem_mma;
#endif
          compute_int_qk<
              num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
              low_qk_inner, (HeadDim == 64 ? SwizzleMode::k64B
                                           : SwizzleMode::k128B),
              HeadDim / kPackInt8, DataType::kInt8>(
              smem_q8, smem_k8, rs_storage.as_int, q_offset, k_offset);
#pragma unroll
          for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; ++fk) {
#pragma unroll
              for (uint32_t element = 0; element < 8; ++element) {
                const float score = __int2float_rz(
                    rs_storage.as_int[fq][fk][element]);
                rs_storage.as_float[fq][fk][element] =
                    score * current_dequant_scale;
              }
            }
          }
#if defined(MPA_K64_BLOCK_MODE)
          const uint32_t valid_k = static_cast<uint32_t>(
              valid_k_counts[batch_id * num_physical_stages +
                             absolute_stage]);
          const uint32_t k_index_base =
              get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
              2 * (lane_id % 4);
          apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(
              k_index_base, rs_storage.as_float, valid_k);
#else
          const uint32_t k_index_base =
              absolute_stage * kCtaK +
              get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
              2 * (lane_id % 4);
          apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(
              k_index_base, rs_storage.as_float, kv_len);
#endif
          update_mdo<
              num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(
              rs_storage.as_float, ro, m, d, base2_softmax_scale);
          accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(
              rs_storage.as_float, d);
          uint32_t rs_fp8[num_tiles_q][num_tiles_k / 2][4];
          RS_32_to_8<num_tiles_q, num_tiles_k>(
              rs_storage.as_float, rs_fp8);

          cp_async::wait_group<0>();
          __syncthreads();
          compute_fp8_sv_inst_buf_fp16_accu<
              num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
              num_tiles_v, SwizzleMode::k64B, kCtaK / kPackFp8>(
              smem_v8, rs_fp8, ro, d);
          __syncthreads();
        }
          leave_fp8_probability_and_v_domains(ro, d, v_scale_base);
    }
#endif
  }
  // Phase: FP16
#if MPA_LOW4_NVFP4 || MPA_MIDDLE_INT8 || MPA_MIDDLE_MXFP8
  const uint32_t high_iterations = HasFp16
      ? static_cast<uint32_t>(
            *reinterpret_cast<volatile int32_t*>(fp16_count + metadata_row))
      : 0;
#else
  const uint32_t high_iterations = initial_high_iterations;
#endif
#if MPA_MIDDLE_INT8 && !MPA_LOW4_NVFP4
  const uint32_t fp16_route_low_iterations = HasFp16
      ? static_cast<uint32_t>(
            *reinterpret_cast<volatile int32_t*>(fp8_count + metadata_row))
      : 0;
#else
  const uint32_t fp16_route_low_iterations = route_low_iterations;
#endif
#if MPA_MIDDLE_INT8 && !MPA_LOW4_NVFP4
  uint32_t fp16_batch_id;
  uint32_t fp16_head_id;
  uint32_t fp16_num_qo_heads;
  if constexpr (HasFp16) {
    asm volatile("mov.u32 %0, %%ctaid.z;" : "=r"(fp16_batch_id));
    asm volatile("mov.u32 %0, %%ctaid.y;" : "=r"(fp16_head_id));
    asm volatile("mov.u32 %0, %%nctaid.y;" : "=r"(fp16_num_qo_heads));
  }
  const uint32_t fp16_q_head_row =
      fp16_batch_id * fp16_num_qo_heads + fp16_head_id;
#else
  const uint32_t fp16_head_id = head_id;
  const uint32_t fp16_num_qo_heads = num_qo_heads;
#endif
  if constexpr (HasFp16) {
    const bool need_q16 = high_iterations != 0;
    HalfQKVSmem<HeadDim> smem_q16(smem);

    constexpr uint32_t half_line_lanes = 8;
    constexpr uint32_t half_lines_per_warp = 4;
    constexpr uint32_t half_smem_iters_row =
        HeadDim / (half_line_lanes * kPackHalf);
    constexpr uint32_t q_half_smem_iters_col =
        kCtaQ / (num_warps * half_lines_per_warp);
    if (need_q16) {
      half* q_lane =
#if MPA_MIDDLE_INT8 && !MPA_LOW4_NVFP4
          q16 + (fp16_q_head_row * qo_len +
#else
          q16 + ((batch_id * fp16_num_qo_heads + fp16_head_id) * qo_len +
#endif
                 query_block * kCtaQ +
                 kCtaQ / num_warps * warp_id + lane_id / half_line_lanes) *
                    HeadDim +
          (lane_id % half_line_lanes) * kPackHalf;
      uint32_t q_smem_load = smem_q16.get_permuted_offset(
          warp_id * half_lines_per_warp * q_half_smem_iters_col +
              lane_id / half_line_lanes,
          lane_id % half_line_lanes);
      const uint32_t q_load_row =
          query_block * kCtaQ + kCtaQ / num_warps * warp_id +
          lane_id / half_line_lanes;
      load_global_to_share<
          half_line_lanes, half_lines_per_warp, half_smem_iters_row,
          q_half_smem_iters_col, SwizzleMode::k128B, HeadDim / kPackHalf,
          kCtaQ>(
          q_lane, q_smem_load, HeadDim, smem_q16, q_load_row, qo_len);
      cp_async::commit_group();
      cp_async::wait_group<0>();
      __syncthreads();
    }

    if constexpr (HasFp8 && SmoothK) {
      if (low_iterations != 0 && high_iterations != 0) {
          const half* mean_base =
              k_mean + (batch_id * num_kv_heads + kv_head) * HeadDim;
          restore_smoothed_low_rowmax<HeadDim, num_tiles_q>(
              smem_q16, mean_base, m, query_block, qo_len,
              base2_softmax_scale);
      }
    }
  }

  if constexpr (HasFp16) {
    HalfQKVSmem<HeadDim> smem_q16(smem);
    HalfQKVSmem<HeadDim> smem_kv16(
        smem + kCtaQ * HeadDim * sizeof(half));

    constexpr uint32_t half_line_lanes = 8;
    constexpr uint32_t half_lines_per_warp = 4;
    constexpr uint32_t half_smem_iters_row =
        HeadDim / (half_line_lanes * kPackHalf);
    constexpr uint32_t kv_half_smem_iters_col =
        kCtaK / (num_warps * half_lines_per_warp);

    if (high_iterations != 0) {
#if MPA_MIDDLE_INT8 && !MPA_LOW4_NVFP4
      const uint32_t fp16_kv_head_row =
          fp16_q_head_row / num_kv_groups;
#else
      const uint32_t fp16_kv_head_row =
          batch_id * num_kv_heads + kv_head;
#endif
      int32_t* high_lut = fp16_lut + metadata_row * num_physical_stages;
      const uint32_t q_smem_mma = smem_q16.get_permuted_offset(
          get_warp_idx_q<num_warps_q, num_warps_k>() * kWarpQ + lane_id % 16,
          lane_id / 16);
      const uint32_t kv_smem_mma = smem_kv16.get_permuted_offset(
          lane_id % 8 + (lane_id / 16) * 8,
          (lane_id / 8) % 2);
      uint32_t absolute_stage = 0;

      for (uint32_t iteration = 0; iteration < high_iterations; ++iteration) {
#if defined(MPA_K64_BLOCK_MODE)
        // The H3 mixed route keeps one compact video list per row. FP8 reads
        // its leading segment. FP16 first executes the exact prefix stages,
        // then reads the following rescue segment after the row's variable
        // FP8 count. This avoids materializing a second full R x K LUT.
        absolute_stage = iteration < fp16_prefix_stages
            ? iteration
            : static_cast<uint32_t>(
                  high_lut[
                      fp16_route_low_iterations + iteration - fp16_prefix_stages]);
#else
        absolute_stage += static_cast<uint32_t>(high_lut[iteration]);
#endif

        if constexpr (HeadDim == 128 && num_tiles_q == 2) {
          if (iteration != 0) {
            if constexpr (HasFp8) {
              reload_d128_fq0_q(
#if MPA_MIDDLE_INT8 && !MPA_LOW4_NVFP4
                  smem_q16, q16, fp16_q_head_row, query_block, qo_len);
#else
                  smem_q16, q16, batch_id, fp16_head_id, fp16_num_qo_heads,
                  query_block, qo_len);
#endif
            }
          }
        }

        if constexpr (HeadDim == 128 && num_tiles_q == 2) {
          const uint32_t k_load_row =
              absolute_stage * kCtaK + kCtaK / num_warps * warp_id +
              lane_id / half_line_lanes;
          uint32_t k_logical_row;
          asm volatile(
              "mad.lo.u32 %0, %1, %2, %3;\n"
              : "=r"(k_logical_row)
              : "r"(fp16_kv_head_row), "r"(kv_len), "r"(k_load_row));
          half* k_lane =
              k16 + k_logical_row * HeadDim +
              (lane_id % half_line_lanes) * kPackHalf;
          uint32_t k_smem_load = smem_kv16.get_permuted_offset(
              warp_id * half_lines_per_warp * kv_half_smem_iters_col +
                  lane_id / half_line_lanes,
              lane_id % half_line_lanes);
          load_global_to_share<
              half_line_lanes, half_lines_per_warp, half_smem_iters_row,
              kv_half_smem_iters_col, SwizzleMode::k128B,
              HeadDim / kPackHalf, kCtaK>(
              k_lane, k_smem_load, HeadDim, smem_kv16, k_load_row, kv_len);
        } else {
          half* k_lane =
              k16 + (fp16_kv_head_row * kv_len +
                     absolute_stage * kCtaK +
                     kCtaK / num_warps * warp_id +
                     lane_id / half_line_lanes) *
                        HeadDim +
              (lane_id % half_line_lanes) * kPackHalf;
          uint32_t k_smem_load = smem_kv16.get_permuted_offset(
              warp_id * half_lines_per_warp * kv_half_smem_iters_col +
                  lane_id / half_line_lanes,
              lane_id % half_line_lanes);
          const uint32_t k_load_row =
              absolute_stage * kCtaK + kCtaK / num_warps * warp_id +
              lane_id / half_line_lanes;
          load_global_to_share<
              half_line_lanes, half_lines_per_warp, half_smem_iters_row,
              kv_half_smem_iters_col, SwizzleMode::k128B,
              HeadDim / kPackHalf, kCtaK>(
              k_lane, k_smem_load, HeadDim, smem_kv16, k_load_row, kv_len);
        }
        cp_async::commit_group();
        cp_async::wait_group<0>();
        __syncthreads();

        if constexpr (HeadDim == 128 && HasFp8 && num_tiles_q == 2) {
          // fq=0 is stashed into its consumed Q rows; fq=1 remains in
          // registers.  This removes 32 live score registers at the QK peak
          // while keeping the CTA-lifetime FP32 m/d/O state in registers.
          float rs_active[1][num_tiles_k][8];
          uint32_t q_offset_fq0 = q_smem_mma;
          uint32_t k_offset_fq0 = kv_smem_mma;
          compute_half_qk<
              num_warps_q, num_warps_k, 1, num_tiles_k,
              half_qk_inner, SwizzleMode::k128B, HeadDim / kPackHalf>(
              smem_q16, smem_kv16, rs_active,
              q_offset_fq0, k_offset_fq0);
          if (
#if defined(MPA_K64_BLOCK_MODE)
              true
#else
              iteration + 1 == high_iterations
#endif
          ) {
#if defined(MPA_K64_BLOCK_MODE)
            const uint32_t valid_k = static_cast<uint32_t>(
                valid_k_counts[batch_id * num_physical_stages +
                               absolute_stage]);
            const uint32_t k_index_base =
                get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
                2 * (lane_id % 4);
            apply_out_of_bound_mask<1, num_tiles_k>(
                k_index_base, rs_active, valid_k);
#else
            const uint32_t k_index_base =
                absolute_stage * kCtaK +
                get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
                2 * (lane_id % 4);
            apply_out_of_bound_mask<1, num_tiles_k>(
                k_index_base, rs_active, kv_len);
#endif
          }
          stash_d128_fq0_scores<num_tiles_k>(smem, rs_active);

          uint32_t q_offset_fq1 = smem_q16.get_permuted_offset(
              get_warp_idx_q<num_warps_q, num_warps_k>() * kWarpQ +
                  kMmaQkM + lane_id % 16,
              lane_id / 16);
          uint32_t k_offset_fq1 = kv_smem_mma;
          compute_half_qk<
              num_warps_q, num_warps_k, 1, num_tiles_k,
              half_qk_inner, SwizzleMode::k128B, HeadDim / kPackHalf>(
              smem_q16, smem_kv16, rs_active,
              q_offset_fq1, k_offset_fq1);
          if (
#if defined(MPA_K64_BLOCK_MODE)
              true
#else
              iteration + 1 == high_iterations
#endif
          ) {
#if defined(MPA_K64_BLOCK_MODE)
            const uint32_t valid_k = static_cast<uint32_t>(
                valid_k_counts[batch_id * num_physical_stages +
                               absolute_stage]);
            const uint32_t k_index_base =
                get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
                2 * (lane_id % 4);
            apply_out_of_bound_mask<1, num_tiles_k>(
                k_index_base, rs_active, valid_k);
#else
            const uint32_t k_index_base =
                absolute_stage * kCtaK +
                get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
                2 * (lane_id % 4);
            apply_out_of_bound_mask<1, num_tiles_k>(
                k_index_base, rs_active, kv_len);
#endif
          }

          // K and V share one 16-KiB scratch region.  V copy overlaps fq=1
          // softmax and P conversion; fq=0 is recovered after fq=1 PV.
          __syncthreads();
          const uint32_t v_load_row =
              absolute_stage * kCtaK + kCtaK / num_warps * warp_id +
              lane_id / half_line_lanes;
          uint32_t v_logical_row;
          // Materialize V's full logical row at its use site.  Leaving the
          // equivalent K/V expression visible lets ptxas keep one row base
          // live across the entire D128 score/softmax/PV region and spill it.
          asm volatile(
              "mad.lo.u32 %0, %1, %2, %3;\n"
              : "=r"(v_logical_row)
              : "r"(fp16_kv_head_row), "r"(kv_len), "r"(v_load_row));
          half* v_lane =
              v16 + v_logical_row * HeadDim +
              (lane_id % half_line_lanes) * kPackHalf;
          uint32_t v_smem_load = smem_kv16.get_permuted_offset(
              warp_id * half_lines_per_warp * kv_half_smem_iters_col +
                  lane_id / half_line_lanes,
              lane_id % half_line_lanes);
          load_global_to_share<
              half_line_lanes, half_lines_per_warp, half_smem_iters_row,
              kv_half_smem_iters_col, SwizzleMode::k128B,
              HeadDim / kPackHalf, kCtaK>(
              v_lane, v_smem_load, HeadDim, smem_kv16, v_load_row, kv_len);
          cp_async::commit_group();

          update_mdo<1, num_tiles_k, num_tiles_v, false, false, false>(
              rs_active, ro + 1, m + 1, d + 1, base2_softmax_scale);
          accumulate_d<1, num_tiles_k, ComputeUnit::kCudaCore>(
              rs_active, d + 1);
          pack_rs_fp16_inplace<1, num_tiles_k>(rs_active);
          cp_async::wait_group<0>();
          __syncthreads();
          compute_fp16_sv_stage_tilewise<
              0, 1, num_tiles_k, num_tiles_v,
              SwizzleMode::k128B, HeadDim / kPackHalf, HasFp8>(
              smem_kv16, rs_active, ro + 1);

          // Reuse the same 32-register fragment for fq=0 instead of exposing
          // two arrays to ptxas in one scope.
          load_d128_fq0_scores<num_tiles_k>(smem, rs_active);
          if constexpr (!HasFp8) {
            if (iteration + 1 < high_iterations) {
              // The fq=0 Q rows are free once their scores are restored.
              // Refill them while the current fq=0 softmax/PV work runs.
              __syncwarp();
              reload_d128_fq0_q<false, true>(
#if MPA_MIDDLE_INT8 && !MPA_LOW4_NVFP4
                  smem_q16, q16, fp16_q_head_row, query_block, qo_len);
#else
                  smem_q16, q16, batch_id, fp16_head_id, fp16_num_qo_heads,
                  query_block, qo_len);
#endif
            }
          }
          update_mdo<1, num_tiles_k, num_tiles_v, false, false, false>(
              rs_active, ro, m, d, base2_softmax_scale);
          accumulate_d<1, num_tiles_k, ComputeUnit::kCudaCore>(rs_active, d);
          pack_rs_fp16_inplace<1, num_tiles_k>(rs_active);
          compute_fp16_sv_stage_tilewise<
              0, 1, num_tiles_k, num_tiles_v,
              SwizzleMode::k128B, HeadDim / kPackHalf, HasFp8>(
              smem_kv16, rs_active, ro);
        } else {
          float rs[num_tiles_q][num_tiles_k][8];
          uint32_t q_offset = q_smem_mma;
          uint32_t k_offset = kv_smem_mma;
          compute_half_qk<
              num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
              half_qk_inner, SwizzleMode::k128B, HeadDim / kPackHalf>(
              smem_q16, smem_kv16, rs, q_offset, k_offset);
          if (
#if defined(MPA_K64_BLOCK_MODE)
              true
#else
              iteration + 1 == high_iterations
#endif
          ) {
#if defined(MPA_K64_BLOCK_MODE)
            const uint32_t valid_k = static_cast<uint32_t>(
                valid_k_counts[batch_id * num_physical_stages +
                               absolute_stage]);
            const uint32_t k_index_base =
                get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
                2 * (lane_id % 4);
            apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(
                k_index_base, rs, valid_k);
#else
            const uint32_t k_index_base =
                absolute_stage * kCtaK +
                get_warp_idx_k<num_warps_q, num_warps_k>() * kWarpK +
                2 * (lane_id % 4);
            apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(
                k_index_base, rs, kv_len);
#endif
          }

          __syncthreads();
          half* v_lane =
              v16 + (fp16_kv_head_row * kv_len +
                     absolute_stage * kCtaK +
                     kCtaK / num_warps * warp_id +
                     lane_id / half_line_lanes) *
                        HeadDim +
              (lane_id % half_line_lanes) * kPackHalf;
          uint32_t v_smem_load = smem_kv16.get_permuted_offset(
              warp_id * half_lines_per_warp * kv_half_smem_iters_col +
                  lane_id / half_line_lanes,
              lane_id % half_line_lanes);
          const uint32_t v_load_row =
              absolute_stage * kCtaK + kCtaK / num_warps * warp_id +
              lane_id / half_line_lanes;
          load_global_to_share<
              half_line_lanes, half_lines_per_warp, half_smem_iters_row,
              kv_half_smem_iters_col, SwizzleMode::k128B,
              HeadDim / kPackHalf, kCtaK>(
              v_lane, v_smem_load, HeadDim, smem_kv16, v_load_row, kv_len);
          cp_async::commit_group();

          update_mdo<
              num_tiles_q, num_tiles_k, num_tiles_v, false, false, false>(
              rs, ro, m, d, base2_softmax_scale);
          accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(
              rs, d);
          pack_rs_fp16_inplace<num_tiles_q, num_tiles_k>(rs);
          cp_async::wait_group<0>();
          __syncthreads();
          compute_fp16_sv_stage_tilewise<
              0, num_tiles_q, num_tiles_k, num_tiles_v,
              SwizzleMode::k128B, HeadDim / kPackHalf>(
              smem_kv16, rs, ro);
        }
        __syncthreads();
      }
    }
  }
