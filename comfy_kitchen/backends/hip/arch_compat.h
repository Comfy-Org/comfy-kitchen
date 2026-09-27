// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Architecture compatibility macros. Defined at compile time based on HIP
// built-in target macros so the kernels do not depend on a generated header.
#pragma once

// RDNA2 (gfx103x) has no matrix cores; the GEMM kernels use VALU v_dot4_i32_i8
// and v_dot2_f32_f16 instead of WMMA. The HIP compiler defines __gfx1035__ etc.
// when compiling for the respective target.
#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || \
    defined(__gfx1033__) || defined(__gfx1034__) || defined(__gfx1035__) || \
    defined(__gfx1036__)
#ifndef __GFX10__
#define __GFX10__ 1
#endif
#endif
