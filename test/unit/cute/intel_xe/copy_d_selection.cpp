/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_xe_2d.hpp>
#include <cute/arch/mma_xe.hpp>

#include "cutlass_unit_test.h"

using namespace cute;

namespace {

using DType = bfloat16_t;
using DpasAtom = MMA_Atom<XE_DPAS_TT<8, float, DType>>;

// Qwen3 small-K shape: WGTile <128,128,16>, SG <4,2>.  Each subgroup owns a
// contiguous 64-column N run (128 / 2), which is two BF16 MMA N-atoms -> the D
// store should be sized to a full 64B cache line (32 elements), not one atom (16).
using SGLayout = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;
using TiledMMA = typename TiledMMAHelper<
    DpasAtom,
    Layout<Shape<_128, _128, _16>>,
    SGLayout>::TiledMMA;

// A tiling whose per-subgroup N run is only one atom wide (tile_n / sg_n = 16):
// the store must stay single-atom (16), i.e. the width is derived, not forced.
using NarrowSGLayout = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;
using NarrowTiledMMA = typename TiledMMAHelper<
    DpasAtom,
    Layout<Shape<_128, _32, _16>>,
    NarrowSGLayout>::TiledMMA;

using RowMajorD = Layout<Shape<_128, _128>, Stride<_128, _1>>;
using ColMajorD = Layout<Shape<_128, _128>, Stride<_1, _128>>;
using RowMajorNarrowD = Layout<Shape<_128, _32>, Stride<_32, _1>>;

template <class Layout>
auto make_d_tensor()
{
  return make_tensor(make_gmem_ptr(static_cast<DType*>(nullptr)), Layout{});
}

using RowMajorCopy = decltype(make_block_2d_copy_D(
    TiledMMA{}, make_d_tensor<RowMajorD>()));
using ColMajorCopy = decltype(make_block_2d_copy_D(
    TiledMMA{}, make_d_tensor<ColMajorD>()));
using NarrowRowMajorCopy = decltype(make_block_2d_copy_D(
    NarrowTiledMMA{}, make_d_tensor<RowMajorNarrowD>()));

// The store width is derived from the per-subgroup output tile's contiguous run, so
// the Qwen small-K shape groups two BF16 MMA N-atoms into one full 64B cache line.
static_assert(RowMajorCopy::CopyOp::AtomWidth == 32,
              "row-major D with a two-atom subgroup N run should fill a full cache line");

// The width follows whichever dimension is gmem-contiguous.  For column-major D the
// contiguous direction is M, and the subgroup's 32-row M run still fills a cache line.
static_assert(ColMajorCopy::CopyOp::AtomWidth == 32,
              "column-major D fills a cache line along its contiguous (M) direction");

// When the per-subgroup contiguous run is a single atom, the width degrades to one
// atom on its own -- the wide store is a property of the derived tiling, never forced.
static_assert(NarrowRowMajorCopy::CopyOp::AtomWidth == 16,
              "a single-atom subgroup run must not be widened");

} // namespace

TEST(PVC_CuTe_Xe, automatic_d_copy_width_is_layout_derived) {
  SUCCEED();
}
