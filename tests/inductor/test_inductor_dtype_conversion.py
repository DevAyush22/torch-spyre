# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EPIC-39 Flexible Data Types — dtype conversion tests.

Organized by coverage area:
  TestExplicitConversion — explicit .to(dtype), shortcuts, non-contiguous layouts
  TestImplicitPromotion  — mixed-dtype tensor ops, reductions, workarounds
  TestDataTransfer       — compiled roundtrips, H2D, D2H, eager cross-device copy
  TestUnsupportedGates   — compiled and eager rejection gates, int64 downcast
  TestModelIntegration   — compiled model-layer integration
"""

import os
import unittest
import warnings

import pytest
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import set_warn_always_context

import utils_inductor
from utils_inductor import cached_randn


class TestExplicitConversion(unittest.TestCase):
    """Bucket B: explicit .to(dtype), shortcut methods, non-contiguous layouts."""

    def compare_with_cpu(self, *args, **kwargs):
        return utils_inductor.compare_with_cpu(*args, **kwargs)

    # ---- Core shape variants ----

    def test_to_dtype_fp16_to_fp32_2d_stick_aligned(self):
        """(4,64) fp16->fp32 roundtrip, 2D stick-aligned shape."""
        x = cached_randn((4, 64), differentiation="sa02_fp16fp32")

        def fn(t):
            return t.to(torch.float32).to(torch.float16)

        self.compare_with_cpu(fn, x, run_eager=False)

    def test_to_dtype_fp16_to_fp32_unaligned_4x16(self):
        """(4,16) fp16->fp32, unaligned last dim."""
        x = cached_randn((4, 16), differentiation="sa03_4x16")

        def fn(t):
            return t.to(torch.float32).to(torch.float16)

        self.compare_with_cpu(fn, x, run_eager=False)

    def test_to_dtype_fp16_to_fp32_unaligned_4x68(self):
        """(4,68) fp16->fp32, unaligned last dim."""
        x = cached_randn((4, 68), differentiation="sa04_4x68")

        def fn(t):
            return t.to(torch.float32).to(torch.float16)

        self.compare_with_cpu(fn, x, run_eager=False)

    def test_to_dtype_fp16_to_fp32_fp32_stick_aligned(self):
        """(4,32) fp16->fp32, fp32-stick-aligned (last dim multiple of 32)."""
        x = cached_randn((4, 32), differentiation="sa05_4x32")

        def fn(t):
            return t.to(torch.float32).to(torch.float16)

        self.compare_with_cpu(fn, x, run_eager=False)

    def test_to_dtype_bf16_to_fp32_2d_stick_aligned(self):
        """(4,64) bf16->fp32 roundtrip, 2D stick-aligned shape."""
        x = cached_randn((4, 64), dtype=torch.bfloat16, differentiation="sa08_bf16fp32")

        def fn(t):
            return t.to(torch.float32).to(torch.bfloat16)

        self.compare_with_cpu(fn, x, run_eager=False)

    def test_to_dtype_chain_fp16_fp32_bf16(self):
        """chain fp16->fp32->bf16."""
        x = cached_randn((4, 64), differentiation="sa12_chain")

        def fn(t):
            return t.to(torch.float32).to(torch.bfloat16)

        self.compare_with_cpu(fn, x, run_eager=False)

    # ---- Dtype-preserving ops after explicit cast ----

    def test_neg_fp16_dtype_preserved_after_explicit_cast(self):
        """neg on fp32-cast of fp16 tensor."""
        x = cached_randn((4, 64), differentiation="sa13_neg")

        def fn(t):
            y = t.to(torch.float32)
            return torch.neg(y).to(torch.float16)

        self.compare_with_cpu(fn, x, run_eager=False)

    def test_abs_fp16_dtype_preserved(self):
        """abs on fp16 preserves fp16 dtype."""
        x = cached_randn((4, 64), differentiation="sa14_abs")

        def fn(t):
            return torch.abs(t)

        self.compare_with_cpu(fn, x, run_eager=False)

    # ---- Multi-core work division ----

    def test_to_dtype_work_division_fp16_fp32_sencores4(self):
        """fp16->fp32 with SENCORES=4 multi-core work division."""
        orig = os.environ.get("SENCORES")
        os.environ["SENCORES"] = "4"
        try:
            x = cached_randn((4, 64), differentiation="wd01_sencores4")

            def fn(t):
                return t.to(torch.float32).to(torch.float16)

            self.compare_with_cpu(fn, x, run_eager=False)
        finally:
            if orig is None:
                os.environ.pop("SENCORES", None)
            else:
                os.environ["SENCORES"] = orig

    # ---- Eager dispatch path ----

    @pytest.mark.skip(reason="eager .to(dtype) on Spyre tensor returns wrong dtype result")
    @pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
    def test_eager_to_dtype_fp32_to_fp16(self):
        """Eager fp32→fp16 cast on a Spyre tensor."""
        x = torch.randn((4, 64), dtype=torch.float32).to("spyre")
        result = x.to(torch.float16)
        assert result.dtype == torch.float16
        assert result.device.type == "spyre"
        expected = x.cpu().to(torch.float16)
        assert torch.allclose(result.cpu(), expected, atol=1e-2, rtol=1e-2)

    # ---- Non-contiguous layouts ----

    def test_to_dtype_transposed_fp16_to_fp32(self):
        """fp16->fp32 on a transposed (non-contiguous) tensor."""
        x = cached_randn((64, 4), differentiation="ncex01_transpose")

        def fn(t):
            return t.t().to(torch.float32).to(torch.float16)

        self.compare_with_cpu(fn, x, run_eager=False)

    @pytest.mark.skip(reason="stride-2 slice generates unsupported stick expression 2*(Mod(d1,32))")
    def test_to_dtype_strided_slice_fp16_to_fp32(self):
        """fp16->fp32 on a stride-2 sliced non-contiguous tensor."""
        x = cached_randn((4, 128), differentiation="ncex02_stride")

        def fn(t):
            return t[:, ::2].to(torch.float32).to(torch.float16)

        self.compare_with_cpu(fn, x, run_eager=False)

    def test_to_dtype_expanded_view_fp16_to_fp32(self):
        """fp16->fp32 on an expand()ed zero-stride tensor."""
        x = cached_randn((1, 64), differentiation="ncex03_expand")

        def fn(t):
            return t.expand(4, 64).to(torch.float32).to(torch.float16)

        self.compare_with_cpu(fn, x, run_eager=False)

    # ---- Shortcut methods .float()/.half()/.bfloat16() ----

    def test_float_shortcut_value_correctness(self):
        """fp16.float() then .half() — compiled roundtrip correctness."""
        x = cached_randn((4, 64), differentiation="sc01_float")

        def fn(t):
            y = t.float()
            return y.half()

        self.compare_with_cpu(fn, x, run_eager=False)

    @pytest.mark.xfail(
        reason=(
            "XFAIL #2205 — fp32.half() uses FP32TODL16_OP; result "
            "stored in DL16_TO_FP32 arrangement; D2H reads garbled values"
        )
    )
    def test_half_shortcut_value_correctness(self):
        """fp32.half() shortcut — FP32TODL16_OP path, XFAIL #2205."""
        x = cached_randn((4, 64), dtype=torch.float32, differentiation="sc02_half")

        def fn(t):
            return t.half()

        self.compare_with_cpu(fn, x, run_eager=False)

    def test_bfloat16_shortcut_from_fp16_value_correctness(self):
        """fp16.bfloat16() uses IDENTITY_OP (same physical storage)."""
        x = cached_randn((4, 64), differentiation="sc03_bf16_from_fp16")

        def fn(t):
            return t.bfloat16()

        self.compare_with_cpu(fn, x, run_eager=False)

    @pytest.mark.xfail(
        reason=(
            "XFAIL #2205 — fp32.bfloat16() uses FP32TODL16_OP; "
            "result stored in DL16_TO_FP32 arrangement; D2H readback garbled"
        )
    )
    def test_bfloat16_shortcut_from_fp32_value_correctness(self):
        """fp32.bfloat16() uses FP32TODL16_OP, XFAIL #2205."""
        x = cached_randn((4, 64), dtype=torch.float32, differentiation="sc04_bf16_from_fp32")

        def fn(t):
            return t.bfloat16()

        self.compare_with_cpu(fn, x, run_eager=False)

    # ---- Special values and edge cases ----

    def test_special_values_nan_inf_preserved_through_fp16_fp32_conversion(self):
        """NaN and Inf must survive fp16->fp32 roundtrip."""
        x = torch.tensor(
            [float("nan"), float("inf"), float("-inf"), 1.0],
            dtype=torch.float16,
        )

        def fn(t):
            return t.to(torch.float32).to(torch.float16)

        self.compare_with_cpu(fn, x, run_eager=False)

    def test_same_dtype_cast_is_noop_in_compiled_graph(self):
        """same-dtype cast (fp16->fp16) is a no-op; regression guard."""
        x = cached_randn((4, 64), differentiation="noop01_fp16")

        def fn(t):
            return t.to(torch.float16)

        self.compare_with_cpu(fn, x, run_eager=False)

    def test_fp32_to_bool_compiled(self):
        """compiled fp32→bool conversion and D2H — both work correctly."""
        x = cached_randn((4, 64), dtype=torch.float32, differentiation="sen05_fp32bool")

        def fn(t):
            return t.to(torch.bool)

        self.compare_with_cpu(fn, x, run_eager=False)

    # ---- Workaround tests: both-fp16 explicit cast (TEMPORARY) ----
    # Both inputs from fp16 → both get DL16_TO_FP32 arrangement → validator accepts.
    # Remove after https://github.com/torch-spyre/torch-spyre/issues/2254 is fixed
    # and the corresponding XFAIL tests below pass directly.

    def test_wa_exp_add_3d_stick_aligned(self):
        """explicit cast WA — 3D (2,4,64) — both DL16_TO_FP32 → PASS."""
        x = cached_randn((2, 4, 64), differentiation="exp3d_x")
        y = cached_randn((2, 4, 64), differentiation="exp3d_y")

        def fn(a, b):
            a_fp32 = a.to(torch.float32)
            b_fp32 = b.to(torch.float32)
            z = torch.add(a_fp32, b_fp32)
            return z.to(torch.float16)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    def test_wa_exp_sub_3d_stick_aligned(self):
        """explicit sub WA — 3D (2,4,64) — both DL16_TO_FP32 → PASS."""
        x = cached_randn((2, 4, 64), differentiation="expsub3d_x")
        y = cached_randn((2, 4, 64), differentiation="expsub3d_y")

        def fn(a, b):
            a_fp32 = a.to(torch.float32)
            b_fp32 = b.to(torch.float32)
            z = torch.sub(a_fp32, b_fp32)
            return z.to(torch.float16)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    def test_wa_exp_mul_4d_stick_aligned(self):
        """explicit mul WA — 4D (2,2,4,64) — both DL16_TO_FP32 → PASS."""
        x = cached_randn((2, 2, 4, 64), differentiation="expmul4d_x")
        y = cached_randn((2, 2, 4, 64), differentiation="expmul4d_y")

        def fn(a, b):
            a_fp32 = a.to(torch.float32)
            b_fp32 = b.to(torch.float32)
            z = torch.mul(a_fp32, b_fp32)
            return z.to(torch.float16)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    def test_wa_exp_add_non_contiguous_2d(self):
        """non-contiguous slice (4,128)→(4,64) — both DL16_TO_FP32 → PASS."""
        x = cached_randn((4, 128), differentiation="expnc_x")
        y = cached_randn((4, 128), differentiation="expnc_y")

        def fn(a, b):
            a_nc = a[:, :64]
            b_nc = b[:, :64]
            a_fp32 = a_nc.to(torch.float32)
            b_fp32 = b_nc.to(torch.float32)
            return torch.add(a_fp32, b_fp32).to(torch.float16)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    # ---- Known failures: cast+chain and unaligned downstream ----

    @pytest.mark.xfail(
        reason=(
            "XFAIL #2254 — add output arrangement differs from "
            "c_fp32 (DL16_TO_FP32); mul rejects mismatched inputs"
        )
    )
    def test_explicit_fp16_to_fp32_chained_add_mul_3d(self):
        """chained add→mul with all-fp16 explicit-cast, 3D aligned."""
        x = cached_randn((2, 4, 64), differentiation="expch_x")
        y = cached_randn((2, 4, 64), differentiation="expch_y")
        z = cached_randn((2, 4, 64), differentiation="expch_z")

        def fn(a, b, c):
            a_fp32 = a.to(torch.float32)
            b_fp32 = b.to(torch.float32)
            c_fp32 = c.to(torch.float32)
            out = torch.mul(torch.add(a_fp32, b_fp32), c_fp32)
            return out.to(torch.float16)

        self.compare_with_cpu(fn, x, y, z, run_eager=False)

    @pytest.mark.xfail(
        reason=(
            "XFAIL #2204 — unaligned last dim (16) causes padding "
            "propagation failure in to_dtype for fp16→fp32"
        )
    )
    def test_explicit_fp16_to_fp32_add_3d_unaligned(self):
        """explicit cast — 3D unaligned (2,4,16)."""
        x = cached_randn((2, 4, 16), differentiation="exp3du_x")
        y = cached_randn((2, 4, 16), differentiation="exp3du_y")

        def fn(a, b):
            a_fp32 = a.to(torch.float32)
            b_fp32 = b.to(torch.float32)
            z = torch.add(a_fp32, b_fp32)
            return z.to(torch.float16)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    @pytest.mark.xfail(
        reason=(
            "XFAIL #2204 — unaligned last dim (16) causes padding "
            "propagation failure in to_dtype for fp16→fp32"
        )
    )
    def test_explicit_fp16_to_fp32_add_4d_unaligned(self):
        """explicit cast — 4D unaligned (2,2,4,16)."""
        x = cached_randn((2, 2, 4, 16), differentiation="exp4du_x")
        y = cached_randn((2, 2, 4, 16), differentiation="exp4du_y")

        def fn(a, b):
            a_fp32 = a.to(torch.float32)
            b_fp32 = b.to(torch.float32)
            z = torch.add(a_fp32, b_fp32)
            return z.to(torch.float16)

        self.compare_with_cpu(fn, x, y, run_eager=False)


class TestImplicitPromotion(unittest.TestCase):
    """Bucket C: mixed-dtype tensor ops, reductions with dtype=, scalar promotion."""

    def compare_with_cpu(self, *args, **kwargs):
        return utils_inductor.compare_with_cpu(*args, **kwargs)

    # ---- Reference: same-arrangement workaround ----
    # Both inputs fp16: Inductor promotes b via to_dtype → both get DL16_TO_FP32.
    # Demonstrates the workaround pattern for #2254.

    def test_implicit_add_fp16_3d_stick_aligned(self):
        """implicit add workaround — 3D (2,4,64) — both DL16_TO_FP32 → PASS."""
        x = cached_randn((2, 4, 64), differentiation="imp3d_x")
        y = cached_randn((2, 4, 64), differentiation="imp3d_y")

        def fn(a, b):
            a_fp32 = a.to(torch.float32)
            z = torch.add(a_fp32, b)
            return z.to(a.dtype)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    # ---- Direct tensor+tensor mixed-dtype (XFAIL #2254) ----
    # H2D fp32 tensor has STANDARD arrangement; on-device fp16→fp32 gives DL16_TO_FP32.
    # Mismatch on the op inputs → rejected by SDSC validator.
    # https://github.com/torch-spyre/torch-spyre/issues/2254

    @pytest.mark.xfail(
        reason=(
            "XFAIL #2254 — explicit fp16→fp32 cast + H2D fp32 tensor: "
            "DL16_TO_FP32 vs STANDARD arrangement mismatch on add inputs"
        )
    )
    def test_working_implicit_add_fp16_to_fp32(self):
        """explicit fp16->fp32 cast still hits #2254 arrangement mismatch."""
        x = cached_randn((4, 64), differentiation="newim01_x")
        y = cached_randn((4, 64), dtype=torch.float32, differentiation="newim01_y")

        def fn(a, b):
            a_fp32 = a.to(torch.float32)
            return torch.add(a_fp32, b)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    @pytest.mark.xfail(
        reason=(
            "XFAIL #2254 — direct fp16 + H2D fp32 tensor: "
            "DL16_TO_FP32 vs STANDARD arrangement mismatch"
        )
    )
    def test_direct_mixed_dtype_add_arrangement_error(self):
        """direct add of H2D fp32 + on-device fp16 hits arrangement mismatch."""
        x = cached_randn((4, 64), differentiation="newim02_fp16")
        y = cached_randn((4, 64), dtype=torch.float32, differentiation="newim02_fp32")

        def fn(a, b):
            return a + b

        self.compare_with_cpu(fn, x, y, run_eager=False)

    @pytest.mark.xfail(
        reason="XFAIL #2254 — direct fp16+fp32 4D add hits arrangement mismatch"
    )
    def test_direct_mixed_dtype_add_4d_stick_aligned(self):
        """direct fp16+fp32 add on 4D (2,2,4,64) without workaround."""
        x = cached_randn((2, 2, 4, 64), differentiation="dir4d_fp16")
        y = cached_randn((2, 2, 4, 64), dtype=torch.float32, differentiation="dir4d_fp32")

        def fn(a, b):
            return torch.add(a, b)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    @pytest.mark.xfail(
        reason="XFAIL #2254 — direct fp16+fp32 5D add hits arrangement mismatch"
    )
    def test_direct_mixed_dtype_add_5d_stick_aligned(self):
        """direct fp16+fp32 add on 5D (2,2,2,4,64) without workaround."""
        x = cached_randn((2, 2, 2, 4, 64), differentiation="dir5d_fp16")
        y = cached_randn((2, 2, 2, 4, 64), dtype=torch.float32, differentiation="dir5d_fp32")

        def fn(a, b):
            return torch.add(a, b)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    @pytest.mark.xfail(
        reason="XFAIL #2254 — direct fp16+fp32 large-value add hits arrangement mismatch"
    )
    def test_direct_mixed_dtype_add_large_values(self):
        """direct fp16+fp32 add with large values (scale=100)."""
        x = cached_randn((4, 64), differentiation="dirlrg_fp16", scale=100.0)
        y = cached_randn((4, 64), dtype=torch.float32, differentiation="dirlrg_fp32", scale=100.0)

        def fn(a, b):
            return torch.add(a, b)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    @pytest.mark.xfail(
        reason="XFAIL #2254 — direct fp16+fp32 small-value add hits arrangement mismatch"
    )
    def test_direct_mixed_dtype_add_small_values(self):
        """direct fp16+fp32 add with small values (scale=1e-3)."""
        x = cached_randn((4, 64), differentiation="dirsml_fp16", scale=1e-3)
        y = cached_randn((4, 64), dtype=torch.float32, differentiation="dirsml_fp32", scale=1e-3)

        def fn(a, b):
            return torch.add(a, b)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    # ---- WA for dimension/value variants (TEMPORARY) ----
    # Both inputs from fp16 → both get DL16_TO_FP32 → same arrangement → PASS.
    # Remove after https://github.com/torch-spyre/torch-spyre/issues/2254 is fixed.

    def test_wa_add_4d_both_fp16(self):
        """4D (2,2,4,64) add — both DL16_TO_FP32 → PASS."""
        x = cached_randn((2, 2, 4, 64), differentiation="wa4d_x")
        y = cached_randn((2, 2, 4, 64), differentiation="wa4d_y")

        def fn(a, b):
            return torch.add(a.to(torch.float32), b.to(torch.float32)).to(torch.float16)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    def test_wa_add_5d_both_fp16(self):
        """5D (2,2,2,4,64) add — both DL16_TO_FP32 → PASS."""
        x = cached_randn((2, 2, 2, 4, 64), differentiation="wa5d_x")
        y = cached_randn((2, 2, 2, 4, 64), differentiation="wa5d_y")

        def fn(a, b):
            return torch.add(a.to(torch.float32), b.to(torch.float32)).to(torch.float16)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    def test_wa_add_large_values_both_fp16(self):
        """large-value add (scale=100) — both DL16_TO_FP32 → PASS."""
        x = cached_randn((4, 64), differentiation="walrg_x", scale=100.0)
        y = cached_randn((4, 64), differentiation="walrg_y", scale=100.0)

        def fn(a, b):
            return torch.add(a.to(torch.float32), b.to(torch.float32)).to(torch.float16)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    def test_wa_add_small_values_both_fp16(self):
        """small-value add (scale=1e-3) — both DL16_TO_FP32 → PASS."""
        x = cached_randn((4, 64), differentiation="wasml_x", scale=1e-3)
        y = cached_randn((4, 64), differentiation="wasml_y", scale=1e-3)

        def fn(a, b):
            return torch.add(a.to(torch.float32), b.to(torch.float32)).to(torch.float16)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    # ---- Unaligned shapes (XFAIL #2204) ----

    @pytest.mark.xfail(
        reason=(
            "XFAIL #2204 — unaligned last dim (32) causes padding "
            "propagation failure in to_dtype for fp16→fp32"
        )
    )
    def test_implicit_add_fp16_2d_unaligned(self):
        """implicit add workaround — 2D unaligned (4,32)."""
        x = cached_randn((4, 32), differentiation="imp2du_x")
        y = cached_randn((4, 32), differentiation="imp2du_y")

        def fn(a, b):
            a_fp32 = a.to(torch.float32)
            z = torch.add(a_fp32, b)
            return z.to(a.dtype)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    @pytest.mark.xfail(
        reason=(
            "XFAIL #2204 — unaligned last dim (16) causes padding "
            "propagation failure in to_dtype for fp16→fp32"
        )
    )
    def test_implicit_add_fp16_3d_unaligned(self):
        """implicit add workaround — 3D unaligned (2,4,16)."""
        x = cached_randn((2, 4, 16), differentiation="imp3du_x")
        y = cached_randn((2, 4, 16), differentiation="imp3du_y")

        def fn(a, b):
            a_fp32 = a.to(torch.float32)
            z = torch.add(a_fp32, b)
            return z.to(a.dtype)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    @pytest.mark.xfail(
        reason=(
            "XFAIL #2204 — unaligned last dim (16) causes padding "
            "propagation failure in to_dtype for fp16→fp32"
        )
    )
    def test_implicit_add_fp16_4d_unaligned(self):
        """implicit add workaround — 4D unaligned (2,2,4,16)."""
        x = cached_randn((2, 2, 4, 16), differentiation="imp4du_x")
        y = cached_randn((2, 2, 4, 16), differentiation="imp4du_y")

        def fn(a, b):
            a_fp32 = a.to(torch.float32)
            z = torch.add(a_fp32, b)
            return z.to(a.dtype)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    # ---- Other binary ops (XFAIL #2254) ----

    @pytest.mark.xfail(
        reason=(
            "XFAIL #2254 — sub with explicit fp16→fp32 cast: "
            "DL16_TO_FP32 vs STANDARD arrangement mismatch"
        )
    )
    def test_sub_fp16_fp32_via_explicit_cast_roundtrip(self):
        """sub with explicit fp16->fp32 cast hits #2254."""
        x = cached_randn((4, 64), differentiation="spe09_x")
        y = cached_randn((4, 64), dtype=torch.float32, differentiation="spe09_y")

        def fn(a, b):
            a_fp32 = a.to(torch.float32)
            return torch.sub(a_fp32, b).to(torch.float16)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    @pytest.mark.xfail(
        reason=(
            "XFAIL #2254 — mul with explicit fp16→fp32 cast: "
            "DL16_TO_FP32 vs STANDARD arrangement mismatch"
        )
    )
    def test_mul_fp16_fp32_via_explicit_cast_roundtrip(self):
        """mul with explicit fp16->fp32 cast hits #2254."""
        x = cached_randn((4, 64), differentiation="spe10_x")
        y = cached_randn((4, 64), dtype=torch.float32, differentiation="spe10_y")

        def fn(a, b):
            a_fp32 = a.to(torch.float32)
            return torch.mul(a_fp32, b).to(torch.float16)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    @pytest.mark.xfail(
        reason=(
            "XFAIL #2254 — div with explicit fp16→fp32 cast: "
            "DL16_TO_FP32 vs STANDARD arrangement mismatch"
        )
    )
    def test_div_fp16_by_fp32_via_explicit_cast_workaround(self):
        """div with explicit fp16->fp32 cast hits #2254."""
        x = cached_randn((4, 64), differentiation="spe12_x")
        y = cached_randn((4, 64), dtype=torch.float32, differentiation="spe12_y")

        def fn(a, b):
            a_fp32 = a.to(torch.float32)
            return torch.div(a_fp32, b).to(torch.float16)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    def test_true_divide_fp16_fp16_always_promotes_to_fp32(self):
        """true_divide(fp16, fp16) must return fp32 per PyTorch semantics."""
        x = cached_randn((4, 64), differentiation="tp01_x")
        y = cached_randn((4, 64), differentiation="tp01_y")

        def fn(a, b):
            return torch.true_divide(a, b)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    @pytest.mark.xfail(
        reason=(
            "XFAIL #2254 — fp32 - fp16 reversed operands: "
            "STANDARD (H2D fp32) vs DL16_TO_FP32 (implicit fp16→fp32) arrangement mismatch"
        )
    )
    def test_direct_sub_fp32_tensor_fp16_tensor(self):
        """fp32_tensor - fp16_tensor (reversed operand order)."""
        x = cached_randn((4, 64), dtype=torch.float32, differentiation="tp02_fp32")
        y = cached_randn((4, 64), differentiation="tp02_fp16")

        def fn(a, b):
            return torch.sub(a, b)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    @pytest.mark.xfail(
        reason=(
            "XFAIL #2254 — stack([fp16→fp32, fp32]) inserts implicit to_dtype; "
            "DL16_TO_FP32 and STANDARD arrangements mismatch on stack"
        )
    )
    def test_stack_fp16_fp32_implicit_promotion(self):
        """stack([fp16, fp32]) implicit promotion."""
        x = cached_randn((4, 64), differentiation="tp04_fp16")
        y = cached_randn((4, 64), dtype=torch.float32, differentiation="tp04_fp32")

        def fn(a, b):
            return torch.stack([a.to(torch.float32), b])

        self.compare_with_cpu(fn, x, y, run_eager=False)

    @pytest.mark.xfail(
        reason=(
            "XFAIL #2254 — cat([fp16→fp32, fp32]) results in "
            "DL16_TO_FP32 + STANDARD arrangement mismatch"
        )
    )
    def test_cat_fp16_fp32_implicit_promotion(self):
        """cat([fp16, fp32]) implicit promotion."""
        x = cached_randn((4, 64), differentiation="spe06_fp16")
        y = cached_randn((4, 64), dtype=torch.float32, differentiation="spe06_fp32")

        def fn(a, b):
            return torch.cat([a.to(torch.float32), b], dim=0)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    # ---- WA for div/stack/cat (TEMPORARY) ----
    # Both inputs fp16 → both DL16_TO_FP32 → same arrangement.
    # Remove passing WA tests after #2254 is fixed.

    def test_wa_div_both_fp16(self):
        """div workaround with compatible DL16_TO_FP32 inputs."""
        x = cached_randn((4, 64), differentiation="wadiv_x")
        # Keep divisors away from zero so FP16 rounding is not artificially amplified.
        y = cached_randn((4, 64), differentiation="wadiv_y", abs=True) + 1.0

        def fn(a, b):
            return torch.div(a.to(torch.float32), b.to(torch.float32)).to(torch.float16)

        self.compare_with_cpu(fn, x, y, atol=1e-2, rtol=1e-2, run_eager=False)

    def test_wa_stack_both_fp16(self):
        """stack — both DL16_TO_FP32 → PASS."""
        x = cached_randn((4, 64), differentiation="wastack_x")
        y = cached_randn((4, 64), differentiation="wastack_y")

        def fn(a, b):
            return torch.stack(
                [a.to(torch.float32), b.to(torch.float32)]
            ).to(torch.float16)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    def test_wa_cat_both_fp16(self):
        """cat dim=0 — both DL16_TO_FP32 → PASS."""
        x = cached_randn((4, 64), differentiation="wacat_x")
        y = cached_randn((4, 64), differentiation="wacat_y")

        def fn(a, b):
            return torch.cat(
                [a.to(torch.float32), b.to(torch.float32)], dim=0
            ).to(torch.float16)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    # ---- Reductions with dtype= ----

    @pytest.mark.skip(reason="SDSC rejects dl16tofp32 fused as a reduction operand")
    def test_sum_fp16_with_dtype_fp32(self):
        """torch.sum(fp16, dtype=fp32) — implicit cast fused into reduction."""
        x = cached_randn((4, 64), differentiation="spe03_sum")

        def fn(t):
            return torch.sum(t, dtype=torch.float32)

        self.compare_with_cpu(fn, x, run_eager=False)

    @pytest.mark.skip(reason="SDSC rejects dl16tofp32 fused as a reduction operand")
    def test_sum_bf16_dtype_fp32(self):
        """torch.sum(bf16, dtype=fp32) — same issue as fp16 variant."""
        x = cached_randn((4, 64), dtype=torch.bfloat16, differentiation="spe03bf16_sum")

        def fn(t):
            return torch.sum(t, dtype=torch.float32)

        self.compare_with_cpu(fn, x, run_eager=False)

    def test_wa_sum_fp16_explicit_fp32(self):
        """t.to(fp32) first, then sum — PASS (TEMPORARY).

        Materializes fp32 buffer before summing so dl16tofp32 is not fused into
        the reduction op. Remove after SDSC codegen issue is fixed.
        """
        x = cached_randn((4, 64), differentiation="spe03wa_sum")

        def fn(t):
            return torch.sum(t.to(torch.float32))

        self.compare_with_cpu(fn, x, run_eager=False)

    def test_wa_sum_bf16_explicit_fp32(self):
        """bf16 WA — t.to(fp32) first, then sum — PASS (TEMPORARY)."""
        x = cached_randn((4, 64), dtype=torch.bfloat16, differentiation="wa_spe03bf16_sum")

        def fn(t):
            return torch.sum(t.to(torch.float32))

        self.compare_with_cpu(fn, x, run_eager=False)

    @pytest.mark.xfail(
        reason=(
            "XFAIL #2205 — mean(fp16, dtype=fp32) decomposes as "
            "to_dtype(fp16→fp32)→sum(fp32)→divide_by_N; fp32 result has "
            "DL16_TO_FP32 arrangement; D2H reads wrong bytes"
        )
    )
    def test_mean_fp16_with_dtype_fp32(self):
        """XFAIL #2205 — mean(fp16, dtype=fp32) D2H arrangement mismatch."""
        x = cached_randn((4, 64), differentiation="spe04_mean")

        def fn(t):
            return torch.mean(t, dtype=torch.float32)

        self.compare_with_cpu(fn, x, run_eager=False)

    # ---- Non-contiguous mixed-dtype ----

    @pytest.mark.xfail(
        reason=(
            "XFAIL #2254 — non-contiguous fp16.t()→fp32 gives DL16_TO_FP32 "
            "while fp32.t() has STANDARD arrangement → mismatch on add"
        )
    )
    def test_implicit_tensor_add_non_contiguous_fp16_fp32(self):
        """non-contiguous fp16 + fp32 hits arrangement mismatch."""
        x = cached_randn((64, 4), differentiation="ncim02_fp16")
        y = cached_randn((64, 4), dtype=torch.float32, differentiation="ncim02_fp32")

        def fn(a, b):
            a_fp32 = a.t().to(torch.float32)
            return torch.add(a_fp32, b.t())

        self.compare_with_cpu(fn, x, y, run_eager=False)

    @pytest.mark.skip(reason="fp16 + 0-dim fp32 tensor triggers IndexError in codegen")
    def test_fp16_plus_0dim_fp32_tensor_implicit_promotion(self):
        """fp16 tensor + 0-dim fp32 tensor implicit promotion."""
        x = cached_randn((4, 64), differentiation="nc_im03_x")

        def fn(a):
            scalar_fp32 = torch.tensor(1.0, dtype=torch.float32, device=a.device)
            return a + scalar_fp32

        self.compare_with_cpu(fn, x, run_eager=False)

    @pytest.mark.skip(reason="transposed fp16 — no mechanism to resolve stick incompatibility for to_dtype on column-major tensor")
    def test_wa_add_transposed_both_fp16(self):
        """add workaround with transposed fp16 inputs — stick incompatibility."""
        x = cached_randn((64, 4), differentiation="wanc_x")
        y = cached_randn((64, 4), differentiation="wanc_y")

        def fn(a, b):
            return torch.add(
                a.t().to(torch.float32), b.t().to(torch.float32)
            ).to(torch.float16)

        self.compare_with_cpu(fn, x, y, run_eager=False)

    # ---- Chained scalar ops and multi-core ----

    @pytest.mark.xfail(
        reason=(
            "XFAIL #2254 — scalar '1.0' materializes as STANDARD fp32 "
            "constant; t.to(fp32) → DL16_TO_FP32; add(DL16_TO_FP32, STANDARD) fails"
        )
    )
    def test_to_dtype_scalar_add_chain_multi_core(self):
        """to_dtype+add(scalar '1.0')+to_dtype chain with SENCORES=4."""
        orig = os.environ.get("SENCORES")
        os.environ["SENCORES"] = "4"
        try:
            x = cached_randn((4, 64), differentiation="wd07_mc")

            def fn(t):
                y = t.to(torch.float32)
                return (y + 1.0).to(torch.float16)

            self.compare_with_cpu(fn, x, run_eager=False)
        finally:
            if orig is None:
                os.environ.pop("SENCORES", None)
            else:
                os.environ["SENCORES"] = orig


class TestDataTransfer(unittest.TestCase):
    """Bucket D (compiled): roundtrip and H2D dtype conversion via Inductor."""

    def compare_with_cpu(self, *args, **kwargs):
        return utils_inductor.compare_with_cpu(*args, **kwargs)

    def test_bf16_compiled_round_trip(self):
        """bf16->fp32->bf16 compiled roundtrip."""
        x = cached_randn((4, 64), dtype=torch.bfloat16, differentiation="sen04_bf16rt")

        def fn(t):
            return t.to(torch.float32).to(torch.bfloat16)

        self.compare_with_cpu(fn, x, run_eager=False)

    def test_roundtrip_cpu_fp32_spyre_fp16_cpu(self):
        """CPU fp32 -> Spyre fp16 compute -> CPU fp32."""
        x = cached_randn((4, 64), dtype=torch.float32, differentiation="ddt05_fp32")

        def fn(t):
            return t.to(torch.float16).to(torch.float32)

        self.compare_with_cpu(fn, x, run_eager=False)

    def test_roundtrip_cpu_fp16_spyre_fp32_cpu(self):
        """CPU fp16 -> Spyre fp32 compute -> CPU fp16."""
        x = cached_randn((4, 64), differentiation="ddt06_fp16")

        def fn(t):
            return t.to(torch.float32).to(torch.float16)

        self.compare_with_cpu(fn, x, run_eager=False)

    def test_roundtrip_cpu_bf16_spyre_fp32_cpu(self):
        """CPU bf16 -> Spyre fp32 compute -> CPU bf16."""
        x = cached_randn((4, 64), dtype=torch.bfloat16, differentiation="ddt07_bf16")

        def fn(t):
            return t.to(torch.float32).to(torch.bfloat16)

        self.compare_with_cpu(fn, x, run_eager=False)

    def test_h2d_with_simultaneous_dtype_conversion(self):
        """tensor.to('spyre', dtype=fp16) in a single call."""
        x_cpu = torch.randn((4, 64), dtype=torch.float32)
        x_spyre = x_cpu.to("spyre", dtype=torch.float16)
        assert x_spyre.device.type == "spyre"
        assert x_spyre.dtype == torch.float16
        expected = x_cpu.to(torch.float16)
        assert torch.allclose(x_spyre.cpu(), expected, atol=1e-2, rtol=1e-2)

    # ---- copy_() compiled path smoke tests ----

    def test_compiled_copy_inplace_fp16(self):
        """compiled same-dtype copy_() for fp16 tensors — PASS.

        Same-dtype (fp16→fp16) compiled copy_() works. The Spyre layout propagation
        pass handles same-dtype copy without cross-format conversion. Contrast with
        """
        src = cached_randn((4, 64), differentiation="cop01_src")

        def fn(t):
            dst = torch.empty_like(t)
            dst.copy_(t)
            return dst

        self.compare_with_cpu(fn, src, run_eager=False)

    def test_compiled_copy_fp16_workaround(self):
        """workaround for compiled copy_() via .to(t.dtype) identity cast.

        copy_() crashes dxp_standalone for fp16 (sdsc_fused_copy node). Using
        .to(t.dtype) instead maps to IDENTITY_OP — no SDSC node is generated,
        so dxp_standalone is not invoked and the test passes. This confirms the
        failure is specific to the sdsc_fused_copy lowering, not to fp16 compiled
        ops in general.
        """
        src = cached_randn((4, 64), differentiation="cop01wa_src")

        def fn(t):
            return t.to(t.dtype)

        self.compare_with_cpu(fn, src, run_eager=False)

    def test_compiled_d2h_fp32_to_fp16(self):
        """compiled D2H fp32 → fp16 in a single .to('cpu', dtype=fp16) call.

        Smoke test confirmed PASS. DeviceCopy warnings from propagate_layouts and
        work_division are non-fatal — the passes skip DeviceCopy nodes but values
        are correct.
        """
        x_cpu = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)
        x_spyre = x_cpu.to("spyre")

        def fn(t):
            return t.to("cpu", dtype=torch.float16)

        compiled = torch.compile(fn, backend="inductor")
        result = compiled(x_spyre)
        expected = x_cpu.to(torch.float16)
        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(result.dtype, torch.float16)
        self.assertTrue(
            torch.allclose(result, expected, atol=1e-2, rtol=1e-2),
            msg=f"D2H fp32→fp16 value mismatch: got {result}, expected {expected}",
        )

    @pytest.mark.skip(reason="cross-dtype copy_() fp32→fp16 — aten.copy.default has no STL candidate for IEEE_FP32→SEN169_FP16")
    def test_compiled_copy_cross_dtype_fp32_to_fp16(self):
        """Compiled copy_() from fp32 into fp16 Spyre tensor — missing cross-format layout candidate."""
        src_fp32 = cached_randn((4, 64), dtype=torch.float32, differentiation="cop02_src")
        dst_fp16 = torch.zeros(4, 64, dtype=torch.float16)

        def fn(d, s):
            d.copy_(s)
            return d

        self.compare_with_cpu(fn, dst_fp16, src_fp32, run_eager=False)

    # ---- Eager cross-device transfer tests ----

    def test_cross_device_copy_with_dtype_coercion_value_check(self):
        """copy_() from fp32 CPU into fp16 Spyre tensor with value check."""
        src_cpu = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        dst_spyre = torch.empty(1, 4, dtype=torch.float16, device="spyre")
        dst_spyre.copy_(src_cpu)
        expected = src_cpu.to(torch.float16)
        self.assertTrue(
            torch.allclose(dst_spyre.cpu(), expected, atol=1e-2, rtol=1e-2),
            msg=f"copy_ value mismatch: got {dst_spyre.cpu()}, expected {expected}",
        )

    @pytest.mark.xfail(
        reason=(
            "XFAIL #2205 #1482 #1497 — .float() on Spyre tensor produces DL16_TO_FP32 "
            "arrangement; direct .cpu() gives wrong values; .half() may also fail"
        ),
        strict=True,
    )
    def test_float_half_methods_on_spyre_tensor(self):
        """.float() and .half() shortcut methods on Spyre tensors, XFAIL #2205."""
        x_cpu = torch.randn(4, 64, dtype=torch.float16)
        x_spyre = x_cpu.to("spyre")
        x_float = x_spyre.float()
        self.assertEqual(x_float.dtype, torch.float32)
        x_back = x_float.half()
        self.assertEqual(x_back.dtype, torch.float16)
        self.assertTrue(
            torch.allclose(x_back.cpu(), x_cpu, atol=1e-2, rtol=1e-2),
            msg="float/half roundtrip values mismatch",
        )

    # ---- Eager H2D dtype quirks ----

    def test_h2d_fp32_to_bool_dtype_conversion(self):
        """fp32→bool H2D succeeds — PyTorch converts dtype on CPU first.

        .to("spyre", dtype=bool) is decomposed: fp32→bool on CPU (values → 0/1),
        then bool→Spyre H2D (bool is stored as fp16, identity pair). So no RuntimeError.
        D2H of bool from Spyre is known broken — xfail #488.
        """
        fp32_cpu = torch.randn(2, 2, dtype=torch.float32)
        result = fp32_cpu.to("spyre", dtype=torch.bool)
        self.assertEqual(result.device.type, "spyre")
        self.assertEqual(result.dtype, torch.bool)

    def test_int64_in_range_value_preserved(self):
        """int64 value 42 round-trips correctly after silent downcast to int32."""
        import torch_spyre  # noqa: F401

        t = torch.tensor([42], dtype=torch.int64)
        spyre_t = t.to("spyre")
        cpu_t = spyre_t.cpu()
        self.assertEqual(cpu_t.dtype, torch.int64)
        self.assertEqual(cpu_t[0].item(), 42)

    def test_int64_out_of_range_truncates_no_exception(self):
        """int64 value 2^40 silently truncates to int32 range without crash."""
        import torch_spyre  # noqa: F401

        large_val = 2**40
        t = torch.tensor([large_val], dtype=torch.int64)
        spyre_t = t.to("spyre")
        cpu_t = spyre_t.cpu()
        self.assertEqual(cpu_t.dtype, torch.int64)
        self.assertNotEqual(
            cpu_t[0].item(), large_val, msg="Expected truncation for 2^40"
        )

    def test_int64_warning_fires_once_per_session(self):
        """int64 downcast warning fires when downcast_warning is enabled.

        Uses set_warn_always_context(True) to bypass the C++ TORCH_WARN_ONCE session
        flag, which may already be set from an earlier test in the same process. The
        "fires at most once per unique call site" guarantee is provided by the C++ layer
        and is tested by test_warning_emitted_when_enabled in test_spyre.py.
        """
        import torch_spyre  # noqa: F401

        torch.spyre.set_downcast_warning(True)
        t = torch.randint(-32768, 32767, (4, 4), dtype=torch.int64)
        with set_warn_always_context(True):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                t.to(device="spyre")
        int64_warnings = [w for w in rec if "int64" in str(w.message)]
        self.assertGreaterEqual(
            len(int64_warnings),
            1,
            msg=f"Expected at least 1 int64 warning, got {len(int64_warnings)}",
        )


class TestUnsupportedGates(unittest.TestCase):
    """Compiled and eager rejection-gate tests for unsupported dtype paths."""

    # ---- Compiled gate tests ----

    def test_add_int8_fp16_raises_error(self):
        """compiled int8+fp16 add raises InductorError (unsupported pair).

        int8+fp16 implicit promotion is not in DtypeOpTable — inductor raises
        InductorError at codegen because no to_dtype lowering exists for int8.
        """
        x = torch.randint(-10, 10, (4, 64), dtype=torch.int8, device="spyre")
        y = cached_randn((4, 64), differentiation="unsup_imp01").to("spyre")

        def fn(a, b):
            return a + b

        compiled = torch.compile(fn, backend="inductor")
        with pytest.raises(torch._inductor.exc.InductorError):
            compiled(x, y)


    # ---- Eager gate tests ----

    def test_h2d_copy_fp64_raises_error(self):
        """fp64 H2D transfer raises RuntimeError (fp64 unsupported on Spyre)."""
        x = torch.randn(2, 2, dtype=torch.float64)
        with self.assertRaises(RuntimeError):
            x.to("spyre")


class TestModelIntegration(unittest.TestCase):
    """Compiled model integration tests: dtype behavior through realistic compute patterns."""

    def compare_with_cpu(self, *args, **kwargs):
        return utils_inductor.compare_with_cpu(*args, **kwargs)

    def test_compiled_linear_fp16(self):
        """compiled fp16 matmul — linear-layer forward pass equivalent."""
        w = cached_randn((32, 64), differentiation="mi01_w")
        x = cached_randn((4, 64), differentiation="mi01_x")

        def forward(inp, weight):
            return torch.mm(inp, weight.t())

        self.compare_with_cpu(forward, x, w, run_eager=False)



if __name__ == "__main__":
    unittest.main()
