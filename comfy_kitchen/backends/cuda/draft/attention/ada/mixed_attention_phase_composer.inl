/*
 * Compile-time SM89 Q64/Q128 ordered precision-phase composition.
 *
 * This file is intentionally included inside the attention kernel after
 * the persistent online-softmax state (ro/m/d) is initialized. Execution is
 * strictly INT8 prefix followed by FP16 suffix, and only ro/m/d is intended
 * to remain live across that boundary. Precision-specific fragments, scales,
 * LUT cursors, metadata-derived values, and async-pipeline state stay in their
 * lexical phase scope, and disabled phases disappear via if constexpr.
 */
  // Phase: INT8 QK with E4M3 V. All copy-pipeline cursors, score fragments,
  // and dequantization scales die before the precision transition below.
  if constexpr (HasFp8) {
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

      const uint32_t q_smem_mma = smem_q8.get_permuted_offset(
          get_warp_idx_q<num_warps_q, num_warps_k>() * kWarpQ + lane_id % 16,
          lane_id / 16);
      const uint32_t k_smem_mma = smem_k8.get_permuted_offset(
          lane_id % 8 + (lane_id / 16) * 8,
          (lane_id / 8) % 2);
      const float q_dequant_scale =
          q_scale[(batch_id * num_qo_heads + head_id) * num_query_blocks +
                  query_block];
      uint32_t absolute_stage = 0;

      // Keep the inherited Sparge/Sage copy/compute pipeline for every INT8
      // phase. K and V occupy disjoint shared-memory regions:
      // with two committed groups outstanding, wait_group<1> exposes the
      // older operand while the newer operand continues to overlap compute.
#if !MPA_DENSE_SEQUENTIAL
        int32_t* low_lut = fp8_lut + metadata_row * num_physical_stages;
        int32_t* low_delta = low_lut;
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
          uint32_t q_offset;
          uint32_t k_offset;
          if constexpr (HasFp16 && kCtaQ == 128) {
            q_offset = smem_q8.get_permuted_offset(
                get_warp_idx_q<num_warps_q, num_warps_k>() * kWarpQ +
                    lane_id % 16,
                lane_id / 16);
            k_offset = smem_k8.get_permuted_offset(
                lane_id % 8 + (lane_id / 16) * 8,
                (lane_id / 8) % 2);
          } else {
            q_offset = q_smem_mma;
            k_offset = k_smem_mma;
          }
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
          uint32_t q_offset;
          uint32_t k_offset;
          if constexpr (HasFp16 && kCtaQ == 128) {
            q_offset = smem_q8.get_permuted_offset(
                get_warp_idx_q<num_warps_q, num_warps_k>() * kWarpQ +
                    lane_id % 16,
                lane_id / 16);
            k_offset = smem_k8.get_permuted_offset(
                lane_id % 8 + (lane_id / 16) * 8,
                (lane_id / 8) % 2);
          } else {
            q_offset = q_smem_mma;
            k_offset = k_smem_mma;
          }
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
          uint32_t q_offset;
          uint32_t k_offset;
          if constexpr (HasFp16 && kCtaQ == 128) {
            q_offset = smem_q8.get_permuted_offset(
                get_warp_idx_q<num_warps_q, num_warps_k>() * kWarpQ +
                    lane_id % 16,
                lane_id / 16);
            k_offset = smem_k8.get_permuted_offset(
                lane_id % 8 + (lane_id / 16) * 8,
                (lane_id / 8) % 2);
          } else {
            q_offset = q_smem_mma;
            k_offset = k_smem_mma;
          }
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
    }
  }

  // Load future-phase metadata at its consuming boundary. In the mixed
  // specialization, keeping this count live through the INT8 mainloop makes
  // ptxas spill persistent softmax state. Pure FP16 keeps its entry load.
  const uint32_t high_iterations = HasFp16
      ? (HasFp8 && kCtaQ == 128
             ? static_cast<uint32_t>(*reinterpret_cast<volatile int32_t*>(
                   fp16_count + metadata_row))
             : initial_high_iterations)
      : 0;

  // Transition: FP16 rescue needs Q for its own MMA, while K-smooth needs the
  // same staged Q only to restore the per-row q*mean offset.  In particular,
  // a pure-INT8 row still needs the latter for a correct LSE even though its
  // normalized output is invariant to the constant shift.
  if constexpr (HasFp16) {
    const bool need_q16 =
        high_iterations != 0 || (SmoothK && low_iterations != 0);
    HalfQKVSmem<HeadDim> smem_q16(smem);

    constexpr uint32_t half_line_lanes = 8;
    constexpr uint32_t half_lines_per_warp = 4;
    constexpr uint32_t half_smem_iters_row =
        HeadDim / (half_line_lanes * kPackHalf);
    constexpr uint32_t q_half_smem_iters_col =
        kCtaQ / (num_warps * half_lines_per_warp);
    if (need_q16) {
      half* q_lane =
          q16 + ((batch_id * num_qo_heads + head_id) * qo_len +
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

    if constexpr (SmoothK) {
      if (low_iterations != 0) {
          const half* mean_base =
              k_mean + (batch_id * num_kv_heads + kv_head) * HeadDim;
          restore_smoothed_low_rowmax<HeadDim, num_tiles_q>(
              smem_q16, mean_base, m, query_block, qo_len,
              base2_softmax_scale);
      }
    }
  }

  // Precision-boundary conversion: every executed FP8 phase leaves both the
  // probability-offset domain and the quantized-V domain before the final
  // normalize, including the compile-time FP8-only family.
  if constexpr (HasFp8) {
    if (low_iterations != 0) {
      const float* v_scale_base =
          v_scale + (batch_id * num_kv_heads + kv_head) * HeadDim;
      leave_fp8_probability_and_v_domains(ro, d, v_scale_base);
    }
  }

  if constexpr (HasFp8 && HasFp16 && kCtaQ == 128) {
    handoff_online_softmax_registers(ro, m, d);
  }

  // Phase: FP16 QK/PV. Its LUT cursor, shared-memory views, and score
  // fragments are disjoint from the INT8 phase-local scope above.
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
      int32_t* high_lut = fp16_lut + metadata_row * num_physical_stages;
#if defined(MPA_K64_BLOCK_MODE) && MPA_CTA_Q == 64
      // Q64 mixed routing stores the rescue segment immediately after the
      // INT8 segment. A monotonic phase-local cursor avoids carrying the
      // low-phase length through every FP16 LUT address; pure FP16 and Q128
      // keep their independently tuned indexed route path below.
      int32_t* high_cursor =
          HasFp8 ? high_lut + low_iterations : nullptr;
#endif
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
#if MPA_CTA_Q == 64
        if constexpr (HasFp8) {
          absolute_stage = iteration < fp16_prefix_stages
              ? iteration
              : static_cast<uint32_t>(*high_cursor++);
        } else {
          absolute_stage = iteration < fp16_prefix_stages
              ? iteration
              : static_cast<uint32_t>(
                    high_lut[
                        low_iterations + iteration - fp16_prefix_stages]);
        }
#else
        absolute_stage = iteration < fp16_prefix_stages
            ? iteration
            : static_cast<uint32_t>(
                  high_lut[
                      low_iterations + iteration - fp16_prefix_stages]);
#endif
#else
        absolute_stage += static_cast<uint32_t>(high_lut[iteration]);
#endif

        if constexpr (HeadDim == 128 && num_tiles_q == 2) {
          const uint32_t k_load_row =
              absolute_stage * kCtaK + kCtaK / num_warps * warp_id +
              lane_id / half_line_lanes;
          const uint32_t kv_head_row =
              batch_id * num_kv_heads + kv_head;
          uint32_t k_logical_row;
          asm volatile(
              "mad.lo.u32 %0, %1, %2, %3;\n"
              : "=r"(k_logical_row)
              : "r"(kv_head_row), "r"(kv_len), "r"(k_load_row));
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
              k16 + ((batch_id * num_kv_heads + kv_head) * kv_len +
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
            v16 + ((batch_id * num_kv_heads + kv_head) * kv_len +
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
        __syncthreads();
      }
    }
  }
