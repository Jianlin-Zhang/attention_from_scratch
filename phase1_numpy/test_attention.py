"""
Test Suite for attention.py
===========================
Run in the project's dedicated conda environment (see environment.yml):

    conda activate attn-scratch                   # Python 3.11 / numpy 1.26.4 / torch 2.2.0 CPU
    python -m unittest test_attention -v          # everything
    python -m unittest test_attention.TestRoPE -v # one topic at a time

Each test compares the pure-NumPy implementation against PyTorch as ground truth,
or against a self-consistency property when PyTorch has no equivalent.
Work through the tests layer by layer — implement the corresponding TODO in
attention.py, then run its test class to validate.

Note on torch 2.2: ``F.rms_norm`` does not exist yet, so the RMSNorm tests build
their own reference and differentiate it with autograd.
"""

import ast
import inspect
import textwrap
import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the module under test
import attention as attn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def torch_to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32)


def np_to_torch(a: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.float32, requires_grad=requires_grad)


# ---------------------------------------------------------------------------
# Numerical gradient checking
# ---------------------------------------------------------------------------
# A second, PyTorch-independent oracle for every backward pass you write.
# A derivative *is* a limit of a difference quotient, so you can approximate any
# gradient with forward passes alone — no derivation required. When an analytic
# gradient disagrees, the report below tells you which element disagrees, which
# usually points straight at the wrong term.
#
# This is also the only oracle available in Phase 2/3: C and CUDA have no
# autograd and no convenient PyTorch to compare against, but central differences
# work in any language.
#
# Recipe for a tensor-valued operator
# -----------------------------------
# ``backward`` computes dL/dθ for a *scalar* L, so invent one: pick a **fixed
# random** ``dout`` and define ``L = sum(y * dout)``. Then dL/dy == dout exactly,
# so ``backward(dout)`` and the numerical estimate refer to the same L.
# Use random values, never all-ones: with dout == 1 some mistakes cancel out and
# the check silently passes.
#
#     lin.W = lin.W.astype(np.float64)          # see the float64 note below
#     x     = rand64(2, 3, 4)
#     dout  = rand64(2, 3, 3)
#
#     y = lin.forward(x); lin.backward(dout)    # analytic
#     analytic = lin.dW.copy()
#
#     numeric = numerical_grad(lambda: float(np.sum(lin.forward(x) * dout)), lin.W)
#     assert_grad_close(analytic, numeric, "dW")

GRAD_CHECK_EPS = 1e-6      # near the float64 sweet spot, eps_machine ** (1/3)
GRAD_CHECK_TOL = 1e-6      # a genuine bug lands at 1e-2 or worse


def rand64(*shape) -> np.ndarray:
    """Random float64 array — the dtype numerical gradients need."""
    return np.random.randn(*shape).astype(np.float64)


def numerical_grad(loss_fn, x: np.ndarray, eps: float = GRAD_CHECK_EPS) -> np.ndarray:
    """Central-difference gradient of ``loss_fn`` w.r.t. every element of ``x``.

    Parameters
    ----------
    loss_fn : zero-argument callable returning a Python float.
              It must **read** ``x`` (directly or through an object holding it);
              this function perturbs ``x`` in place and calls ``loss_fn`` again.
              Taking no arguments is deliberate: it makes the helper work
              identically for inputs and for layer parameters.
    x : the array to differentiate w.r.t. Modified in place and restored, so the
        caller sees it unchanged on return.
    eps : perturbation size.

    Returns
    -------
    grad : same shape as ``x``, dtype float64.

    Why float64 is required
    -----------------------
    Central differences trade two error sources against each other: truncation
    error O(eps^2), which wants eps small, and catastrophic cancellation when
    subtracting two nearly equal numbers, which wants eps large. The total is
    U-shaped with its minimum near ``eps_machine ** (1/3)``.

        float64: eps_machine 2.2e-16 -> best eps ~6e-6, achievable error ~1e-11
        float32: eps_machine 1.2e-7  -> best eps ~5e-3, achievable error ~1e-5

    In float32 the check is barely usable and collapses entirely for eps < 1e-6,
    so this helper refuses anything else rather than let you spend an afternoon
    doubting a correct derivation.

    Cost
    ----
    Two forward passes per element. Use tiny shapes — correctness bugs show up
    at any size.
    """
    if x.dtype != np.float64:
        raise TypeError(
            f"numerical_grad needs a float64 array, got {x.dtype}. Central "
            f"differences lose almost all precision in float32; upcast the "
            f"inputs and the layer parameters first (see rand64)."
        )

    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = loss_fn()
        x[idx] = orig - eps
        f_minus = loss_fn()
        x[idx] = orig                      # restore before moving on
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
    return grad


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    """Max elementwise relative error.

    Relative rather than absolute because gradient magnitudes vary by orders of
    magnitude. The denominator floor keeps ``0 vs 0`` from becoming ``0/0``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.maximum(np.abs(a) + np.abs(b), 1e-12)
    return float(np.max(np.abs(a - b) / denom))


def assert_grad_close(
    analytic: np.ndarray,
    numeric: np.ndarray,
    name: str = "grad",
    tol: float = GRAD_CHECK_TOL,
) -> float:
    """Compare an analytic gradient against a numerical one.

    Raises ``AssertionError`` with the worst-offending index and both values
    there — that locality is the whole point, since it usually identifies which
    term of the derivation is wrong.

    Returns the relative error so callers can log it.

    Rough scale (float64): < 1e-9 excellent, < 1e-6 fine, > 1e-3 almost
    certainly a real bug rather than floating-point noise.
    """
    analytic = np.asarray(analytic, dtype=np.float64)
    numeric = np.asarray(numeric, dtype=np.float64)

    if analytic.shape != numeric.shape:
        raise AssertionError(
            f"{name}: shape mismatch — analytic {analytic.shape} vs "
            f"numeric {numeric.shape}. A gradient always has the shape of the "
            f"thing it differentiates, so fix the shape before the values."
        )

    err = rel_error(analytic, numeric)
    if not err < tol:
        denom = np.maximum(np.abs(analytic) + np.abs(numeric), 1e-12)
        worst = np.unravel_index(np.argmax(np.abs(analytic - numeric) / denom), analytic.shape)
        raise AssertionError(
            f"{name}: max relative error {err:.3e} exceeds tol {tol:.1e}\n"
            f"  worst element at index {worst}\n"
            f"    analytic = {analytic[worst]!r}\n"
            f"    numeric  = {numeric[worst]!r}\n"
            f"  ratio analytic/numeric = "
            f"{analytic[worst] / numeric[worst] if numeric[worst] != 0 else float('nan'):.6f}\n"
            f"  (a clean ratio like 2.0, -1.0 or 1/(1-p) points at a missing "
            f"factor; scattered noise points at a wrong contraction axis)"
        )
    return err


# ---------------------------------------------------------------------------
# The gradient checker checking itself
# ---------------------------------------------------------------------------
# Before trusting a tool to validate your derivations, validate the tool. These
# use functions whose derivatives are known by hand, plus one deliberately wrong
# gradient to prove the checker can actually fail.

class TestGradientChecker(unittest.TestCase):
    def test_cubic(self):
        """f(x) = sum(x^3), so df/dx = 3x^2."""
        x = np.array([[1.3, -0.7], [2.0, 0.5]], dtype=np.float64)
        numeric = numerical_grad(lambda: float(np.sum(x ** 3)), x)
        assert_grad_close(3.0 * x ** 2, numeric, "d(sum x^3)/dx", tol=1e-8)

    def test_transcendental(self):
        """f(x) = sum(sin(x) * exp(x/2)) — product and chain rule together.

        Uses the default (looser) tolerance on purpose: at x = -1.1 the analytic
        derivative nearly vanishes, and relative error divides by a near-zero
        denominator, so it inflates to ~1e-8 even though the absolute agreement is
        ~1e-10. Expect this whenever a gradient element passes through zero — it
        is a property of the metric, not a bug in the derivation.
        """
        x = np.array([0.3, -1.1, 2.2], dtype=np.float64)
        numeric = numerical_grad(lambda: float(np.sum(np.sin(x) * np.exp(x / 2))), x)
        analytic = np.exp(x / 2) * (np.cos(x) + 0.5 * np.sin(x))
        assert_grad_close(analytic, numeric, "d(sum sin*exp)/dx")

    def test_matmul_loss_pattern(self):
        """The exact pattern used for real layers: L = sum((x @ A) * dout)."""
        np.random.seed(70)
        x, A, dout = rand64(3, 4), rand64(4, 2), rand64(3, 2)
        numeric = numerical_grad(lambda: float(np.sum((x @ A) * dout)), x)
        assert_grad_close(dout @ A.T, numeric, "dL/dx", tol=1e-8)

    def test_detects_a_missing_factor(self):
        """A wrong analytic gradient must be rejected, or the tool is useless."""
        x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        numeric = numerical_grad(lambda: float(np.sum(x ** 2)), x)
        with self.assertRaises(AssertionError):
            assert_grad_close(x, numeric, "deliberately missing factor 2")

    def test_detects_a_wrong_axis(self):
        """Right magnitude, wrong layout — the transposed-gradient bug class."""
        np.random.seed(71)
        x, A, dout = rand64(4, 4), rand64(4, 4), rand64(4, 4)
        numeric = numerical_grad(lambda: float(np.sum((x @ A) * dout)), x)
        with self.assertRaises(AssertionError):
            assert_grad_close((dout @ A.T).T, numeric, "transposed on purpose")

    def test_reports_shape_mismatch_first(self):
        with self.assertRaises(AssertionError) as cm:
            assert_grad_close(np.zeros((2, 3)), np.zeros((3, 2)), "mismatched")
        self.assertIn("shape mismatch", str(cm.exception))

    def test_input_is_restored(self):
        """Perturbations must not leak: the caller's array comes back unchanged."""
        x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        before = x.copy()
        numerical_grad(lambda: float(np.sum(x ** 2)), x)
        np.testing.assert_array_equal(x, before)

    def test_float32_is_refused(self):
        x = np.array([1.0, 2.0], dtype=np.float32)
        with self.assertRaises(TypeError):
            numerical_grad(lambda: float(np.sum(x ** 2)), x)

    def test_rel_error_edges(self):
        a = np.array([1.0, -2.0, 0.0])
        self.assertEqual(rel_error(a, a), 0.0)
        self.assertAlmostEqual(rel_error(np.array([1.0]), np.array([-1.0])), 1.0, places=9)
        self.assertEqual(rel_error(np.zeros(3), np.zeros(3)), 0.0)


class TestLinearGradientCheck(unittest.TestCase):
    """Cross-check Linear with the second oracle.

    Linear already passes against PyTorch, so agreement here validates the
    *checker*. These three methods double as copy-paste templates for every
    backward still to be written — dx, then one block per parameter.
    """

    def setUp(self):
        set_seed(72)
        self.lin = attn.Linear(4, 3)
        # Upcast the parameters: numerical differentiation needs float64 end to end
        self.lin.W = self.lin.W.astype(np.float64)
        self.lin.b = self.lin.b.astype(np.float64)
        self.x = rand64(2, 3, 4)
        self.dout = rand64(2, 3, 3)

    def _analytic(self):
        """Fresh forward + backward; returns (dx, dW, db)."""
        self.lin.forward(self.x)
        dx = self.lin.backward(self.dout)
        return dx, self.lin.dW.copy(), self.lin.db.copy()

    def _loss(self) -> float:
        return float(np.sum(self.lin.forward(self.x) * self.dout))

    def test_dx(self):
        dx, _, _ = self._analytic()
        assert_grad_close(dx, numerical_grad(self._loss, self.x), "dx")

    def test_dW(self):
        _, dW, _ = self._analytic()
        assert_grad_close(dW, numerical_grad(self._loss, self.lin.W), "dW")

    def test_db(self):
        _, _, db = self._analytic()
        assert_grad_close(db, numerical_grad(self._loss, self.lin.b), "db")


class TestSoftmaxGradientCheck(unittest.TestCase):
    """A coupled (non-diagonal) Jacobian, checked without PyTorch."""

    def test_softmax_backward(self):
        np.random.seed(73)
        x, dout = rand64(3, 5), rand64(3, 5)
        analytic = attn.softmax_backward(dout, attn.softmax(x, axis=-1), axis=-1)
        numeric = numerical_grad(lambda: float(np.sum(attn.softmax(x, axis=-1) * dout)), x)
        assert_grad_close(analytic, numeric, "dx of softmax")


class TestDropoutGradientCheck(unittest.TestCase):
    """How to gradient-check a *stochastic* operator.

    Two forward passes with different masks are two different functions, so the
    difference quotient would be meaningless. Reseeding immediately before every
    forward pass freezes the mask and makes the operator deterministic.
    """

    def test_dropout_backward_with_frozen_mask(self):
        np.random.seed(74)
        x, dout = rand64(6, 6), rand64(6, 6)
        p, seed = 0.5, 1234

        def loss() -> float:
            np.random.seed(seed)                      # same mask every call
            out, _ = attn.dropout_forward(x, p=p, training=True)
            return float(np.sum(out * dout))

        np.random.seed(seed)
        _, mask = attn.dropout_forward(x, p=p, training=True)
        analytic = attn.dropout_backward(dout, mask, p)

        assert_grad_close(analytic, numerical_grad(loss, x), "dx of dropout")


# ---------------------------------------------------------------------------
# Layer 0 — Utilities
# ---------------------------------------------------------------------------

class TestSoftmax(unittest.TestCase):
    def test_basic(self):
        set_seed(0)
        x_np = np.random.randn(3, 5).astype(np.float32)
        x_th = torch.tensor(x_np, requires_grad=True)

        out_np = attn.softmax(x_np, axis=-1)
        out_th = F.softmax(x_th, dim=-1)

        np.testing.assert_allclose(out_np, torch_to_np(out_th), atol=1e-6)

    def test_numerical_stability(self):
        """Large values should not produce NaN."""
        x_np = np.array([[1000.0, 2000.0, 3000.0]], dtype=np.float32)
        out = attn.softmax(x_np, axis=-1)
        self.assertFalse(np.any(np.isnan(out)))
        self.assertAlmostEqual(float(np.sum(out)), 1.0, places=5)

    def test_backward(self):
        set_seed(1)
        x_np = np.random.randn(2, 4).astype(np.float32)
        x_th = torch.tensor(x_np, requires_grad=True)

        out_np = attn.softmax(x_np, axis=-1)
        out_th = F.softmax(x_th, dim=-1)

        dout_np = np.random.randn(2, 4).astype(np.float32)
        dout_th = torch.tensor(dout_np)

        # numpy backward
        dx_np = attn.softmax_backward(dout_np, out_np, axis=-1)

        # torch backward
        out_th.backward(dout_th)
        dx_th = torch_to_np(x_th.grad)

        np.testing.assert_allclose(dx_np, dx_th, atol=1e-6)


class TestCausalMask(unittest.TestCase):
    def test_shape_and_values(self):
        mask = attn.create_causal_mask(4)
        self.assertEqual(mask.shape, (4, 4))
        self.assertEqual(mask.dtype, np.float32)

        # Lower triangle (incl. diag) should be 0
        for i in range(4):
            for j in range(4):
                if j <= i:
                    self.assertEqual(mask[i, j], 0.0, f"({i},{j}) should be 0")
                else:
                    self.assertEqual(mask[i, j], -np.inf, f"({i},{j}) should be -inf")

    def test_attention_with_causal_mask(self):
        """Softmax over a causal-masked row should ignore future positions."""
        mask = attn.create_causal_mask(4)
        scores = np.ones((4, 4), dtype=np.float32)
        masked = scores + mask
        weights = attn.softmax(masked, axis=-1)

        # Row 0: only position 0 is visible → weight = 1.0
        np.testing.assert_allclose(weights[0], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
        # Row 1: positions 0,1 visible → each gets 0.5
        np.testing.assert_allclose(weights[1], [0.5, 0.5, 0.0, 0.0], atol=1e-6)
        # Row 2: positions 0,1,2 → each gets 1/3
        np.testing.assert_allclose(weights[2], [1 / 3, 1 / 3, 1 / 3, 0.0], atol=1e-6)


class TestPaddingMask(unittest.TestCase):
    def test_shape_and_values(self):
        valid_lens = np.array([2, 3], dtype=np.int32)
        mask = attn.create_padding_mask(valid_lens, max_len=4)
        self.assertEqual(mask.shape, (2, 1, 1, 4))

        # Batch 0: positions 0,1 valid; 2,3 masked
        np.testing.assert_equal(mask[0, 0, 0], [0.0, 0.0, -np.inf, -np.inf])
        # Batch 1: positions 0,1,2 valid; 3 masked
        np.testing.assert_equal(mask[1, 0, 0], [0.0, 0.0, 0.0, -np.inf])


# ---------------------------------------------------------------------------
# Layer 1 — Linear
# ---------------------------------------------------------------------------

class TestLinear(unittest.TestCase):
    def setUp(self):
        set_seed(2)
        self.in_features = 8
        self.out_features = 4
        self.batch = 3
        self.seq = 5

        # Our layer
        self.np_linear = attn.Linear(self.in_features, self.out_features)
        # PyTorch equivalent
        self.th_linear = nn.Linear(self.in_features, self.out_features, bias=True)

        # Copy weights
        self.th_linear.weight.data = torch.tensor(self.np_linear.W.T.copy(), dtype=torch.float32)
        self.th_linear.bias.data = torch.tensor(self.np_linear.b.copy(), dtype=torch.float32)

    def test_forward(self):
        x_np = np.random.randn(self.batch, self.seq, self.in_features).astype(np.float32)
        x_th = torch.tensor(x_np, requires_grad=True)

        out_np = self.np_linear.forward(x_np)
        out_th = self.th_linear(x_th)

        np.testing.assert_allclose(out_np, torch_to_np(out_th), atol=1e-5)

    def test_backward(self):
        x_np = np.random.randn(self.batch, self.seq, self.in_features).astype(np.float32)
        x_th = torch.tensor(x_np, requires_grad=True)

        out_np = self.np_linear.forward(x_np)
        out_th = self.th_linear(x_th)

        dout_np = np.random.randn(*out_np.shape).astype(np.float32)
        dout_th = torch.tensor(dout_np)

        dx_np = self.np_linear.backward(dout_np)
        out_th.backward(dout_th)

        # Shapes first: a gradient always has the shape of what it differentiates.
        # assert_allclose broadcasts, so it would silently accept (1, out) for db —
        # that bug would only surface much later, in the MHA tests.
        self.assertEqual(dx_np.shape, x_np.shape)
        self.assertEqual(self.np_linear.dW.shape, self.np_linear.W.shape)
        self.assertEqual(self.np_linear.db.shape, self.np_linear.b.shape)

        # Check input gradients
        np.testing.assert_allclose(dx_np, torch_to_np(x_th.grad), atol=1e-5)
        # Check weight gradients
        np.testing.assert_allclose(self.np_linear.dW, torch_to_np(self.th_linear.weight.grad.T), atol=1e-5)
        # Check bias gradients
        np.testing.assert_allclose(self.np_linear.db, torch_to_np(self.th_linear.bias.grad), atol=1e-5)


# ---------------------------------------------------------------------------
# Layer 2 — Scaled Dot-Product Attention
# ---------------------------------------------------------------------------

class TestScaledDotProductAttention(unittest.TestCase):
    def setUp(self):
        set_seed(3)
        self.batch = 2
        self.seq_q = 4
        self.seq_k = 6
        self.d_k = 8
        self.d_v = 8

        self.Q = np.random.randn(self.batch, self.seq_q, self.d_k).astype(np.float32)
        self.K = np.random.randn(self.batch, self.seq_k, self.d_k).astype(np.float32)
        self.V = np.random.randn(self.batch, self.seq_k, self.d_v).astype(np.float32)

    def test_forward_output(self):
        out, cache = attn.scaled_dot_product_attention_forward(self.Q, self.K, self.V)
        self.assertEqual(out.shape, (self.batch, self.seq_q, self.d_v))

    def test_forward_vs_torch(self):
        out_np, _ = attn.scaled_dot_product_attention_forward(self.Q, self.K, self.V)

        Q_th = torch.tensor(self.Q)
        K_th = torch.tensor(self.K)
        V_th = torch.tensor(self.V)
        out_th = F.scaled_dot_product_attention(Q_th, K_th, V_th, is_causal=False)

        np.testing.assert_allclose(out_np, torch_to_np(out_th), atol=1e-5)

    def test_forward_with_mask(self):
        mask = np.triu(np.ones((self.seq_q, self.seq_k), dtype=np.float32) * -np.inf, k=1)
        mask = mask[np.newaxis, :, :]  # (1, seq_q, seq_k)

        out_np, _ = attn.scaled_dot_product_attention_forward(self.Q, self.K, self.V, mask)
        Q_th = torch.tensor(self.Q)
        K_th = torch.tensor(self.K)
        V_th = torch.tensor(self.V)
        mask_th = torch.tensor(mask)
        out_th = F.scaled_dot_product_attention(Q_th, K_th, V_th, attn_mask=mask_th)

        np.testing.assert_allclose(out_np, torch_to_np(out_th), atol=1e-5)

    def test_backward_vs_torch(self):
        out_np, cache = attn.scaled_dot_product_attention_forward(self.Q, self.K, self.V)
        dout_np = np.random.randn(*out_np.shape).astype(np.float32)

        dQ_np, dK_np, dV_np = attn.scaled_dot_product_attention_backward(dout_np, cache)

        # PyTorch reference
        Q_th = torch.tensor(self.Q, requires_grad=True)
        K_th = torch.tensor(self.K, requires_grad=True)
        V_th = torch.tensor(self.V, requires_grad=True)
        dout_th = torch.tensor(dout_np)

        out_th = F.scaled_dot_product_attention(Q_th, K_th, V_th, is_causal=False)
        out_th.backward(dout_th)

        np.testing.assert_allclose(dQ_np, torch_to_np(Q_th.grad), atol=1e-5)
        np.testing.assert_allclose(dK_np, torch_to_np(K_th.grad), atol=1e-5)
        np.testing.assert_allclose(dV_np, torch_to_np(V_th.grad), atol=1e-5)


# ---------------------------------------------------------------------------
# Layer 3 — Multi-Head Attention
# ---------------------------------------------------------------------------

class _MHAFixture:
    """Shared setup for MHA tests."""
    def setUp(self):
        set_seed(4)
        self.d_model = 16
        self.n_heads = 4
        self.batch = 2
        self.seq_q = 5
        self.seq_k = 5
        self.dropout = 0.0
        self.head_dim = self.d_model // self.n_heads

        # Our MHA
        self.np_mha = attn.MultiHeadAttention(self.d_model, self.n_heads, self.dropout)
        # PyTorch MHA
        self.th_mha = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=self.dropout,
            bias=True,  # PyTorch defaults to True — our Linear uses bias
            batch_first=True,
        )

        # Copy weights from PyTorch to our implementation
        self.np_mha.set_weights_from_torch(self.th_mha)


class TestMultiHeadAttentionSelf(_MHAFixture, unittest.TestCase):
    def test_forward(self):
        x_np = np.random.randn(self.batch, self.seq_q, self.d_model).astype(np.float32)
        x_th = torch.tensor(x_np)

        out_np = self.np_mha.forward(x_np, x_np, x_np, training=False)
        out_th, _ = self.th_mha(x_th, x_th, x_th, need_weights=False)

        np.testing.assert_allclose(out_np, torch_to_np(out_th), atol=1e-5)

    def test_backward(self):
        x_np = np.random.randn(self.batch, self.seq_q, self.d_model).astype(np.float32)
        x_th = torch.tensor(x_np, requires_grad=True)

        # Forward
        out_np = self.np_mha.forward(x_np, x_np, x_np, training=False)
        out_th, _ = self.th_mha(x_th, x_th, x_th, need_weights=False)

        dout_np = np.random.randn(*out_np.shape).astype(np.float32)
        dout_th = torch.tensor(dout_np)

        dQ_np, dK_np, dV_np = self.np_mha.backward(dout_np)
        out_th.backward(dout_th)

        # Compare gradients on the input (same for Q, K, V in self-attention)
        # PyTorch accumulates gradients from all three paths: dQ + dK + dV
        np.testing.assert_allclose(dQ_np + dK_np + dV_np, torch_to_np(x_th.grad), atol=1e-5)

        # Compare weight gradients — PyTorch's in_proj_weight is a single
        # combined matrix, so we concatenate our dW and compare against it.
        np_dW_in = np.concatenate([self.np_mha.W_q.dW, self.np_mha.W_k.dW, self.np_mha.W_v.dW], axis=1)
        # Our dW is (in_features, out_features); PyTorch stores (out_features, in_features)
        np.testing.assert_allclose(np_dW_in, torch_to_np(self.th_mha.in_proj_weight.grad.T), atol=1e-5)

        # Bias gradients
        np_db_in = np.concatenate([self.np_mha.W_q.db, self.np_mha.W_k.db, self.np_mha.W_v.db])
        np.testing.assert_allclose(np_db_in, torch_to_np(self.th_mha.in_proj_bias.grad), atol=1e-5)

        # Output projection weight gradients
        np.testing.assert_allclose(
            self.np_mha.W_o.dW,
            torch_to_np(self.th_mha.out_proj.weight.grad.T),
            atol=1e-5,
        )


class TestMultiHeadAttentionCross(_MHAFixture, unittest.TestCase):
    def test_forward(self):
        """Cross-attention: Q from decoder, K/V from encoder."""
        Q_np = np.random.randn(self.batch, self.seq_q, self.d_model).astype(np.float32)
        K_np = np.random.randn(self.batch, self.seq_k + 2, self.d_model).astype(np.float32)
        V_np = K_np.copy()

        Q_th = torch.tensor(Q_np)
        K_th = torch.tensor(K_np)
        V_th = torch.tensor(V_np)

        out_np = self.np_mha.forward(Q_np, K_np, V_np, training=False)
        out_th, _ = self.th_mha(Q_th, K_th, V_th, need_weights=False)

        np.testing.assert_allclose(out_np, torch_to_np(out_th), atol=1e-5)

    def test_backward(self):
        Q_np = np.random.randn(self.batch, self.seq_q, self.d_model).astype(np.float32)
        K_np = np.random.randn(self.batch, self.seq_k + 2, self.d_model).astype(np.float32)
        V_np = K_np.copy()

        Q_th = torch.tensor(Q_np, requires_grad=True)
        K_th = torch.tensor(K_np, requires_grad=True)
        V_th = torch.tensor(V_np, requires_grad=True)

        out_np = self.np_mha.forward(Q_np, K_np, V_np, training=False)
        out_th, _ = self.th_mha(Q_th, K_th, V_th, need_weights=False)

        dout_np = np.random.randn(*out_np.shape).astype(np.float32)
        dout_th = torch.tensor(dout_np)

        dQ_np, dK_np, dV_np = self.np_mha.backward(dout_np)
        out_th.backward(dout_th)

        np.testing.assert_allclose(dQ_np, torch_to_np(Q_th.grad), atol=1e-5)
        np.testing.assert_allclose(dK_np, torch_to_np(K_th.grad), atol=1e-5)
        np.testing.assert_allclose(dV_np, torch_to_np(V_th.grad), atol=1e-5)


class TestMultiHeadAttentionCausal(_MHAFixture, unittest.TestCase):
    def test_causal_mask(self):
        """Causal self-attention: position i can only attend to positions ≤ i."""
        x_np = np.random.randn(self.batch, self.seq_q, self.d_model).astype(np.float32)
        x_th = torch.tensor(x_np)

        causal_mask = attn.create_causal_mask(self.seq_q)  # (seq, seq)

        out_np = self.np_mha.forward(x_np, x_np, x_np, mask=causal_mask, training=False)
        out_th, _ = self.th_mha(x_th, x_th, x_th, attn_mask=torch.tensor(causal_mask), is_causal=False, need_weights=False)

        np.testing.assert_allclose(out_np, torch_to_np(out_th), atol=1e-5)

    def test_causal_autoregressive(self):
        """Verify that output at position i does not depend on inputs at positions > i."""
        x_np = np.random.randn(self.batch, self.seq_q, self.d_model).astype(np.float32)
        causal_mask = attn.create_causal_mask(self.seq_q)

        # Original output
        out_orig = self.np_mha.forward(x_np, x_np, x_np, mask=causal_mask, training=False)

        # Perturb future positions (position 3 perturbed for all sequences)
        x_perturbed = x_np.copy()
        x_perturbed[:, 3, :] += 100.0

        out_pert = self.np_mha.forward(x_perturbed, x_perturbed, x_perturbed, mask=causal_mask, training=False)

        # Positions 0,1,2 should not change
        np.testing.assert_allclose(out_orig[:, :3, :], out_pert[:, :3, :], atol=1e-5)
        # Position 4 may change (it can attend to position 3)
        self.assertFalse(np.allclose(out_orig[:, 4, :], out_pert[:, 4, :], atol=1e-5))


class TestMultiHeadAttentionPadding(_MHAFixture, unittest.TestCase):
    def test_padding_mask(self):
        """Attention with a padding mask should ignore masked key positions."""
        x_np = np.random.randn(self.batch, self.seq_q, self.d_model).astype(np.float32)

        # Mask the last 2 key positions for all samples
        pad_mask = np.zeros((self.batch, self.seq_q), dtype=np.float32)
        pad_mask[:, -2:] = -np.inf
        pad_mask = pad_mask[:, np.newaxis, :]  # (batch, 1, seq_k)

        # Pad mask should be safe: softmax(-inf) = 0
        out_np = self.np_mha.forward(x_np, x_np, x_np, mask=pad_mask, training=False)
        self.assertFalse(np.any(np.isnan(out_np)))
        self.assertEqual(out_np.shape, (self.batch, self.seq_q, self.d_model))


# ---------------------------------------------------------------------------
# Layer 4 — Layer Normalization
# ---------------------------------------------------------------------------

class TestLayerNorm(unittest.TestCase):
    def setUp(self):
        set_seed(5)
        self.batch = 2
        self.seq = 4
        self.d_model = 16

        self.np_ln = attn.LayerNorm(self.d_model)
        self.th_ln = nn.LayerNorm(self.d_model, eps=self.np_ln.eps)

        # Copy weights
        self.th_ln.weight.data = torch.tensor(self.np_ln.gamma.copy(), dtype=torch.float32)
        self.th_ln.bias.data = torch.tensor(self.np_ln.beta.copy(), dtype=torch.float32)

    def test_forward(self):
        x_np = np.random.randn(self.batch, self.seq, self.d_model).astype(np.float32)
        x_th = torch.tensor(x_np, requires_grad=True)

        out_np = self.np_ln.forward(x_np)
        out_th = self.th_ln(x_th)

        np.testing.assert_allclose(out_np, torch_to_np(out_th), atol=1e-5)

    def test_backward(self):
        x_np = np.random.randn(self.batch, self.seq, self.d_model).astype(np.float32)
        x_th = torch.tensor(x_np, requires_grad=True)

        out_np = self.np_ln.forward(x_np)
        out_th = self.th_ln(x_th)

        dout_np = np.random.randn(*out_np.shape).astype(np.float32)
        dout_th = torch.tensor(dout_np)

        dx_np = self.np_ln.backward(dout_np)
        out_th.backward(dout_th)

        np.testing.assert_allclose(dx_np, torch_to_np(x_th.grad), atol=1e-5)
        np.testing.assert_allclose(self.np_ln.dgamma, torch_to_np(self.th_ln.weight.grad), atol=1e-5)
        np.testing.assert_allclose(self.np_ln.dbeta, torch_to_np(self.th_ln.bias.grad), atol=1e-5)


# ---------------------------------------------------------------------------
# Layer 5 — Attention Block (bonus)
# ---------------------------------------------------------------------------

class TestAttentionBlock(unittest.TestCase):
    def setUp(self):
        set_seed(6)
        self.batch = 2
        self.seq = 4
        self.d_model = 16
        self.n_heads = 4
        self.d_ff = 32

        self.block = attn.AttentionBlock(self.d_model, self.n_heads, self.d_ff, dropout=0.0)

    def test_forward_shape(self):
        x_np = np.random.randn(self.batch, self.seq, self.d_model).astype(np.float32)
        out = self.block.forward(x_np, training=False)
        self.assertEqual(out.shape, (self.batch, self.seq, self.d_model))

    def test_forward_no_nan(self):
        x_np = np.random.randn(self.batch, self.seq, self.d_model).astype(np.float32)
        out = self.block.forward(x_np, training=False)
        self.assertFalse(np.any(np.isnan(out)))


# ---------------------------------------------------------------------------
# Layer 0 — Dropout  (was missing from the original suite)
# ---------------------------------------------------------------------------

class TestDropout(unittest.TestCase):
    def test_eval_mode_is_identity(self):
        x = np.random.randn(4, 6).astype(np.float32)
        out, mask = attn.dropout_forward(x, p=0.5, training=False)
        np.testing.assert_array_equal(out, x)
        self.assertIsNone(mask)

    def test_p_zero_is_identity(self):
        x = np.random.randn(4, 6).astype(np.float32)
        out, mask = attn.dropout_forward(x, p=0.0, training=True)
        np.testing.assert_array_equal(out, x)
        self.assertIsNone(mask)

    def test_kept_units_are_rescaled(self):
        """Surviving activations must be exactly x/(1-p); dropped ones exactly 0."""
        set_seed(11)
        p = 0.5
        x = np.ones((200, 200), dtype=np.float32)
        out, mask = attn.dropout_forward(x, p=p, training=True)
        self.assertIsNotNone(mask)
        uniq = np.unique(np.round(out, 5))
        self.assertTrue(set(uniq.tolist()).issubset({0.0, round(1.0 / (1 - p), 5)}))

    def test_expectation_preserved(self):
        """Inverted dropout keeps E[out] == x, which is why no rescale is needed at eval."""
        set_seed(12)
        p = 0.3
        x = np.ones((500, 500), dtype=np.float32)
        out, _ = attn.dropout_forward(x, p=p, training=True)
        self.assertAlmostEqual(float(out.mean()), 1.0, delta=0.02)

    def test_backward_matches_forward_mask(self):
        """Gradient flows only where the unit survived, with the same 1/(1-p) scale.

        Note the deliberately non-uniform ``dout``: an all-ones upstream gradient
        would make ``dout * mask / (1-p)`` numerically equal to ``mask / (1-p)``,
        so an implementation that forgets to multiply by ``dout`` at all would
        still pass. Backward returns a vector-Jacobian product, not the Jacobian.
        """
        set_seed(13)
        p = 0.4
        x = np.random.randn(50, 50).astype(np.float32)
        out, mask = attn.dropout_forward(x, p=p, training=True)

        dout = np.random.randn(50, 50).astype(np.float32) * 3.0
        dx = attn.dropout_backward(dout, mask, p)

        np.testing.assert_allclose(dx, dout * mask / (1 - p), atol=1e-6)
        # Dropped positions must be exactly zero, survivors exactly dout/(1-p)
        dropped = mask == 0.0
        np.testing.assert_array_equal(dx[dropped], 0.0)
        np.testing.assert_allclose(dx[~dropped], dout[~dropped] / (1 - p), atol=1e-6)

    def test_backward_is_linear_in_dout(self):
        """Scaling the upstream gradient must scale the result by the same factor."""
        set_seed(14)
        p = 0.3
        x = np.random.randn(20, 20).astype(np.float32)
        _, mask = attn.dropout_forward(x, p=p, training=True)

        dout = np.random.randn(20, 20).astype(np.float32)
        dx1 = attn.dropout_backward(dout, mask, p)
        dx2 = attn.dropout_backward(dout * 7.0, mask, p)

        np.testing.assert_allclose(dx2, dx1 * 7.0, atol=1e-5)

    def test_backward_eval_mode(self):
        dout = np.random.randn(3, 3).astype(np.float32)
        np.testing.assert_array_equal(attn.dropout_backward(dout, None, 0.5), dout)


# ---------------------------------------------------------------------------
# Layer 0b — Explicit-loop kernels (Phase 2/3 bridge)
# ---------------------------------------------------------------------------

def _uses_builtin_matmul(fn) -> bool:
    """Detect '@', np.matmul/dot/einsum/tensordot/inner in a function's own source."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    banned = {"matmul", "dot", "einsum", "tensordot", "inner", "vdot"}
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            return True
        if isinstance(node, ast.Attribute) and node.attr in banned:
            return True
        if isinstance(node, ast.Name) and node.id in banned:
            return True
    return False


class TestMatmulNaive(unittest.TestCase):
    def test_no_builtin_matmul_used(self):
        """The point of this exercise is the loop nest, so shortcuts are disallowed."""
        self.assertFalse(
            _uses_builtin_matmul(attn.matmul_naive),
            "matmul_naive must use explicit scalar loops — no @, matmul, dot, einsum",
        )

    def test_square(self):
        set_seed(20)
        A = np.random.randn(16, 16).astype(np.float32)
        B = np.random.randn(16, 16).astype(np.float32)
        np.testing.assert_allclose(attn.matmul_naive(A, B), A @ B, atol=1e-4)

    def test_rectangular(self):
        set_seed(21)
        A = np.random.randn(7, 13).astype(np.float32)
        B = np.random.randn(13, 5).astype(np.float32)
        np.testing.assert_allclose(attn.matmul_naive(A, B), A @ B, atol=1e-4)

    def test_degenerate_k_of_one(self):
        A = np.random.randn(4, 1).astype(np.float32)
        B = np.random.randn(1, 6).astype(np.float32)
        np.testing.assert_allclose(attn.matmul_naive(A, B), A @ B, atol=1e-5)


class TestMatmulTiled(unittest.TestCase):
    def test_tile_divides_evenly(self):
        set_seed(22)
        A = np.random.randn(64, 32).astype(np.float32)
        B = np.random.randn(32, 16).astype(np.float32)
        np.testing.assert_allclose(attn.matmul_tiled(A, B, tile=16), A @ B, atol=1e-4)

    def test_ragged_edge_tiles(self):
        """M, N, K all indivisible by the tile — the CUDA boundary-check case."""
        set_seed(23)
        A = np.random.randn(37, 29).astype(np.float32)
        B = np.random.randn(29, 23).astype(np.float32)
        np.testing.assert_allclose(attn.matmul_tiled(A, B, tile=8), A @ B, atol=1e-4)

    def test_tile_larger_than_matrix(self):
        set_seed(24)
        A = np.random.randn(5, 5).astype(np.float32)
        B = np.random.randn(5, 5).astype(np.float32)
        np.testing.assert_allclose(attn.matmul_tiled(A, B, tile=128), A @ B, atol=1e-4)

    def test_tile_size_does_not_change_result(self):
        set_seed(25)
        A = np.random.randn(40, 40).astype(np.float32)
        B = np.random.randn(40, 40).astype(np.float32)
        ref = attn.matmul_tiled(A, B, tile=4)
        for t in (7, 16, 33):
            np.testing.assert_allclose(attn.matmul_tiled(A, B, tile=t), ref, atol=1e-4)


class TestOnlineSoftmax(unittest.TestCase):
    """Stream a score row in blocks; the result must equal one-shot softmax @ V."""

    def _run(self, seq_q, seq_k, d_v, block_k):
        set_seed(26)
        scores = np.random.randn(seq_q, seq_k).astype(np.float32) * 3.0
        V = np.random.randn(seq_k, d_v).astype(np.float32)

        m = np.full((seq_q, 1), -np.inf, dtype=np.float32)
        l = np.zeros((seq_q, 1), dtype=np.float32)
        acc = np.zeros((seq_q, d_v), dtype=np.float32)
        for start in range(0, seq_k, block_k):
            stop = min(start + block_k, seq_k)
            m, l, acc = attn.online_softmax_update(m, l, acc, scores[:, start:stop], V[start:stop])

        got = acc / l
        ref = torch_to_np(F.softmax(torch.tensor(scores), dim=-1) @ torch.tensor(V))
        np.testing.assert_allclose(got, ref, atol=1e-5)

    def test_single_block(self):
        self._run(4, 16, 8, block_k=16)

    def test_many_blocks(self):
        self._run(4, 16, 8, block_k=4)

    def test_ragged_last_block(self):
        self._run(3, 17, 8, block_k=5)

    def test_extreme_magnitudes(self):
        """A block whose scores dwarf everything seen so far must not overflow."""
        seq_q, d_v = 2, 4
        scores = np.array([[0.0, 1.0, 900.0, 901.0], [5.0, 5.0, 5.0, 5.0]], dtype=np.float32)
        V = np.arange(4 * d_v, dtype=np.float32).reshape(4, d_v)

        m = np.full((seq_q, 1), -np.inf, dtype=np.float32)
        l = np.zeros((seq_q, 1), dtype=np.float32)
        acc = np.zeros((seq_q, d_v), dtype=np.float32)
        for start in (0, 2):
            m, l, acc = attn.online_softmax_update(m, l, acc, scores[:, start:start + 2], V[start:start + 2])

        got = acc / l
        self.assertFalse(np.any(np.isnan(got)))
        ref = torch_to_np(F.softmax(torch.tensor(scores), dim=-1) @ torch.tensor(V))
        np.testing.assert_allclose(got, ref, atol=1e-4)

    def test_fully_masked_block(self):
        """A block of all -inf (fully masked tile) must be a no-op, not a NaN factory."""
        seq_q, d_v = 2, 3
        good = np.zeros((seq_q, 2), dtype=np.float32)
        dead = np.full((seq_q, 2), -np.inf, dtype=np.float32)
        V = np.ones((2, d_v), dtype=np.float32)

        m = np.full((seq_q, 1), -np.inf, dtype=np.float32)
        l = np.zeros((seq_q, 1), dtype=np.float32)
        acc = np.zeros((seq_q, d_v), dtype=np.float32)
        m, l, acc = attn.online_softmax_update(m, l, acc, good, V)
        m2, l2, acc2 = attn.online_softmax_update(m, l, acc, dead, V * 7.0)

        np.testing.assert_allclose(l2, l, atol=1e-6)
        np.testing.assert_allclose(acc2, acc, atol=1e-6)


# ---------------------------------------------------------------------------
# Layer 0c — Activations
# ---------------------------------------------------------------------------

class TestGELU(unittest.TestCase):
    def setUp(self):
        set_seed(30)
        self.x = np.random.randn(4, 9).astype(np.float32) * 2.0
        self.dout = np.random.randn(4, 9).astype(np.float32)

    def _check(self, mode, torch_mode):
        out, cache = attn.gelu_forward(self.x, approximate=mode)
        x_th = torch.tensor(self.x, requires_grad=True)
        out_th = F.gelu(x_th, approximate=torch_mode)
        np.testing.assert_allclose(out, torch_to_np(out_th), atol=1e-5)

        dx = attn.gelu_backward(self.dout, cache)
        out_th.backward(torch.tensor(self.dout))
        np.testing.assert_allclose(dx, torch_to_np(x_th.grad), atol=1e-5)

    def test_tanh_approximation(self):
        self._check("tanh", "tanh")

    def test_exact_erf(self):
        self._check("none", "none")

    def test_two_modes_are_close_but_not_equal(self):
        a, _ = attn.gelu_forward(self.x, approximate="tanh")
        b, _ = attn.gelu_forward(self.x, approximate="none")
        self.assertLess(float(np.abs(a - b).max()), 1e-2)
        self.assertGreater(float(np.abs(a - b).max()), 0.0)

    def test_non_monotonic_region(self):
        """GELU dips below zero then comes back; the gradient must change sign."""
        x = np.array([-1.5, -0.75, -0.2], dtype=np.float32)
        _, cache = attn.gelu_forward(x, approximate="none")
        g = attn.gelu_backward(np.ones_like(x), cache)
        self.assertLess(g[0], 0.0)
        self.assertGreater(g[2], 0.0)


class TestSiLU(unittest.TestCase):
    def test_forward_backward(self):
        set_seed(31)
        x = np.random.randn(5, 7).astype(np.float32) * 3.0
        dout = np.random.randn(5, 7).astype(np.float32)

        out, cache = attn.silu_forward(x)
        x_th = torch.tensor(x, requires_grad=True)
        out_th = F.silu(x_th)
        np.testing.assert_allclose(out, torch_to_np(out_th), atol=1e-6)

        dx = attn.silu_backward(dout, cache)
        out_th.backward(torch.tensor(dout))
        np.testing.assert_allclose(dx, torch_to_np(x_th.grad), atol=1e-6)

    def test_no_overflow_for_large_negative(self):
        x = np.array([-100.0, -1e4], dtype=np.float32)
        out, _ = attn.silu_forward(x)
        self.assertFalse(np.any(np.isnan(out)))
        np.testing.assert_allclose(out, [0.0, 0.0], atol=1e-6)


class TestSwiGLU(unittest.TestCase):
    def test_forward_backward(self):
        set_seed(32)
        g = np.random.randn(3, 6, 8).astype(np.float32)
        u = np.random.randn(3, 6, 8).astype(np.float32)
        dout = np.random.randn(3, 6, 8).astype(np.float32)

        out, cache = attn.swiglu_forward(g, u)
        g_th = torch.tensor(g, requires_grad=True)
        u_th = torch.tensor(u, requires_grad=True)
        out_th = F.silu(g_th) * u_th
        np.testing.assert_allclose(out, torch_to_np(out_th), atol=1e-6)

        dg, du = attn.swiglu_backward(dout, cache)
        out_th.backward(torch.tensor(dout))
        np.testing.assert_allclose(dg, torch_to_np(g_th.grad), atol=1e-6)
        np.testing.assert_allclose(du, torch_to_np(u_th.grad), atol=1e-6)


# ---------------------------------------------------------------------------
# Layer 0d — Sliding window mask
# ---------------------------------------------------------------------------

class TestSlidingWindowMask(unittest.TestCase):
    def test_pattern(self):
        mask = attn.create_sliding_window_mask(6, window_size=3)
        self.assertEqual(mask.shape, (6, 6))
        self.assertEqual(mask.dtype, np.float32)
        for i in range(6):
            for j in range(6):
                allowed = (j <= i) and (i - j < 3)
                expected = 0.0 if allowed else -np.inf
                self.assertEqual(mask[i, j], expected, f"({i},{j})")

    def test_window_one_is_diagonal(self):
        mask = attn.create_sliding_window_mask(5, window_size=1)
        np.testing.assert_array_equal(np.isfinite(mask), np.eye(5, dtype=bool))

    def test_large_window_equals_causal(self):
        np.testing.assert_array_equal(
            attn.create_sliding_window_mask(5, window_size=99),
            attn.create_causal_mask(5),
        )

    def test_no_row_is_fully_masked(self):
        mask = attn.create_sliding_window_mask(8, window_size=2)
        self.assertTrue(np.all(np.isfinite(mask).any(axis=-1)))
        w = attn.softmax(mask, axis=-1)
        self.assertFalse(np.any(np.isnan(w)))


# ---------------------------------------------------------------------------
# Layer 1 — Embedding
# ---------------------------------------------------------------------------

class TestEmbedding(unittest.TestCase):
    def setUp(self):
        set_seed(33)
        self.vocab, self.dim = 11, 5
        self.emb = attn.Embedding(self.vocab, self.dim)
        self.th_emb = nn.Embedding(self.vocab, self.dim)
        self.th_emb.weight.data = torch.tensor(self.emb.W.copy())
        # Deliberately repeat token 3 — duplicates are the interesting case
        self.idx = np.array([[3, 1, 3], [0, 3, 7]], dtype=np.int64)

    def test_forward(self):
        out = self.emb.forward(self.idx)
        out_th = self.th_emb(torch.tensor(self.idx))
        self.assertEqual(out.shape, (2, 3, self.dim))
        np.testing.assert_allclose(out, torch_to_np(out_th), atol=1e-6)

    def test_backward_scatter_add_with_duplicates(self):
        out = self.emb.forward(self.idx)
        dout = np.random.randn(*out.shape).astype(np.float32)

        self.emb.backward(dout)
        out_th = self.th_emb(torch.tensor(self.idx))
        out_th.backward(torch.tensor(dout))

        self.assertEqual(self.emb.dW.shape, self.emb.W.shape)
        np.testing.assert_allclose(self.emb.dW, torch_to_np(self.th_emb.weight.grad), atol=1e-5)
        # Token 3 appears three times: its row must be the sum of three contributions
        np.testing.assert_allclose(
            self.emb.dW[3], dout[0, 0] + dout[0, 2] + dout[1, 1], atol=1e-5
        )

    def test_unused_rows_get_zero_grad(self):
        out = self.emb.forward(self.idx)
        self.emb.backward(np.ones_like(out))
        for row in (2, 4, 5, 6, 8, 9, 10):
            np.testing.assert_allclose(self.emb.dW[row], 0.0, atol=1e-7)

    def test_padding_idx(self):
        set_seed(34)
        emb = attn.Embedding(self.vocab, self.dim, padding_idx=0)
        np.testing.assert_allclose(emb.W[0], 0.0, atol=1e-7)
        idx = np.array([[0, 1, 0]], dtype=np.int64)
        out = emb.forward(idx)
        emb.backward(np.ones_like(out))
        np.testing.assert_allclose(emb.dW[0], 0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# Layer 3 — GQA / MQA head expansion
# ---------------------------------------------------------------------------

class TestRepeatKV(unittest.TestCase):
    def test_shape(self):
        x = np.random.randn(2, 4, 6, 8).astype(np.float32)
        self.assertEqual(attn.repeat_kv(x, 3).shape, (2, 12, 6, 8))

    def test_n_rep_one_is_identity(self):
        x = np.random.randn(2, 4, 6, 8).astype(np.float32)
        np.testing.assert_array_equal(attn.repeat_kv(x, 1), x)

    def test_grouping_is_contiguous_not_interleaved(self):
        """KV head h must land at output heads [h*n_rep, (h+1)*n_rep)."""
        batch, n_kv, seq, dim, n_rep = 1, 3, 2, 4, 2
        x = np.arange(batch * n_kv * seq * dim, dtype=np.float32).reshape(batch, n_kv, seq, dim)
        out = attn.repeat_kv(x, n_rep)
        for h in range(n_kv):
            for r in range(n_rep):
                np.testing.assert_array_equal(out[:, h * n_rep + r], x[:, h])

    def test_backward_sums_over_group(self):
        batch, n_kv, seq, dim, n_rep = 2, 3, 4, 5, 2
        dout = np.random.randn(batch, n_kv * n_rep, seq, dim).astype(np.float32)
        dx = attn.repeat_kv_backward(dout, n_rep)
        self.assertEqual(dx.shape, (batch, n_kv, seq, dim))
        for h in range(n_kv):
            np.testing.assert_allclose(
                dx[:, h], dout[:, h * n_rep:(h + 1) * n_rep].sum(axis=1), atol=1e-5
            )

    def test_mqa_extreme(self):
        """n_kv_heads == 1 is multi-query attention."""
        x = np.random.randn(2, 1, 3, 4).astype(np.float32)
        out = attn.repeat_kv(x, 8)
        self.assertEqual(out.shape, (2, 8, 3, 4))
        for h in range(8):
            np.testing.assert_array_equal(out[:, h], x[:, 0])


# ---------------------------------------------------------------------------
# Layer 4 — RMSNorm
# ---------------------------------------------------------------------------

def _rms_norm_ref(x_th: torch.Tensor, gamma_th: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference RMSNorm — F.rms_norm does not exist in torch 2.2."""
    var = x_th.pow(2).mean(dim=-1, keepdim=True)
    return x_th * torch.rsqrt(var + eps) * gamma_th


class TestRMSNorm(unittest.TestCase):
    def setUp(self):
        set_seed(35)
        self.d = 12
        self.ln = attn.RMSNorm(self.d, eps=1e-6)
        # Non-trivial gamma, otherwise dgamma bugs hide
        self.ln.gamma = np.random.randn(self.d).astype(np.float32) * 0.5 + 1.0
        self.x = np.random.randn(2, 3, self.d).astype(np.float32) * 2.0

    def test_forward(self):
        out = self.ln.forward(self.x)
        x_th = torch.tensor(self.x)
        g_th = torch.tensor(self.ln.gamma)
        np.testing.assert_allclose(out, torch_to_np(_rms_norm_ref(x_th, g_th, self.ln.eps)), atol=1e-5)

    def test_no_mean_subtraction(self):
        """Unlike LayerNorm, adding a constant to a row changes the output."""
        a = self.ln.forward(self.x)
        b = self.ln.forward(self.x + 5.0)
        self.assertFalse(np.allclose(a, b, atol=1e-3))

    def test_backward(self):
        out = self.ln.forward(self.x)
        dout = np.random.randn(*out.shape).astype(np.float32)

        dx = self.ln.backward(dout)

        x_th = torch.tensor(self.x, requires_grad=True)
        g_th = torch.tensor(self.ln.gamma, requires_grad=True)
        _rms_norm_ref(x_th, g_th, self.ln.eps).backward(torch.tensor(dout))

        np.testing.assert_allclose(dx, torch_to_np(x_th.grad), atol=1e-5)
        np.testing.assert_allclose(self.ln.dgamma, torch_to_np(g_th.grad), atol=1e-5)


# ---------------------------------------------------------------------------
# Layer 6 — Positional information
# ---------------------------------------------------------------------------

def _ref_rope_tables(head_dim: int, max_seq_len: int, base: float = 10000.0):
    half = head_dim // 2
    inv_freq = base ** (-np.arange(half, dtype=np.float64) * 2.0 / head_dim)
    ang = np.arange(max_seq_len, dtype=np.float64)[:, None] * inv_freq[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def _ref_rope_apply_torch(x_th, cos, sin, offset=0):
    """Split-half (GPT-NeoX / HF LLaMA) rotary application, differentiable."""
    seq = x_th.shape[2]
    half = x_th.shape[-1] // 2
    c = torch.tensor(cos[offset:offset + seq])[None, None, :, :]
    s = torch.tensor(sin[offset:offset + seq])[None, None, :, :]
    x1, x2 = x_th[..., :half], x_th[..., half:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class TestSinusoidalPositionalEncoding(unittest.TestCase):
    def test_values(self):
        seq_len, d_model = 7, 8
        pe = attn.sinusoidal_positional_encoding(seq_len, d_model)
        self.assertEqual(pe.shape, (seq_len, d_model))
        self.assertEqual(pe.dtype, np.float32)

        pos = np.arange(seq_len)[:, None]
        i = np.arange(d_model // 2)[None, :]
        ang = pos / (10000.0 ** (2.0 * i / d_model))
        ref = np.zeros((seq_len, d_model), dtype=np.float32)
        ref[:, 0::2] = np.sin(ang)
        ref[:, 1::2] = np.cos(ang)
        np.testing.assert_allclose(pe, ref, atol=1e-6)

    def test_position_zero(self):
        pe = attn.sinusoidal_positional_encoding(3, 6)
        np.testing.assert_allclose(pe[0, 0::2], 0.0, atol=1e-6)
        np.testing.assert_allclose(pe[0, 1::2], 1.0, atol=1e-6)

    def test_bounded(self):
        pe = attn.sinusoidal_positional_encoding(64, 32)
        self.assertLessEqual(float(np.abs(pe).max()), 1.0 + 1e-6)


class TestRoPE(unittest.TestCase):
    def setUp(self):
        set_seed(36)
        self.head_dim = 8
        self.max_len = 16
        self.cos, self.sin = _ref_rope_tables(self.head_dim, self.max_len)
        self.x = np.random.randn(2, 3, 5, self.head_dim).astype(np.float32)

    def test_precompute_tables(self):
        cos, sin = attn.rope_precompute_freqs(self.head_dim, self.max_len)
        self.assertEqual(cos.shape, (self.max_len, self.head_dim // 2))
        self.assertEqual(sin.shape, (self.max_len, self.head_dim // 2))
        np.testing.assert_allclose(cos, self.cos, atol=1e-5)
        np.testing.assert_allclose(sin, self.sin, atol=1e-5)

    def test_position_zero_is_identity(self):
        x = self.x[:, :, :1, :]
        np.testing.assert_allclose(attn.rope_apply(x, self.cos, self.sin, offset=0), x, atol=1e-6)

    def test_forward_split_half_convention(self):
        out = attn.rope_apply(self.x, self.cos, self.sin)
        ref = _ref_rope_apply_torch(torch.tensor(self.x), self.cos, self.sin)
        np.testing.assert_allclose(out, torch_to_np(ref), atol=1e-5)

    def test_forward_with_offset(self):
        """Decoding path: one token sitting at absolute position `offset`."""
        x = self.x[:, :, :1, :]
        for offset in (0, 1, 7, 11):
            out = attn.rope_apply(x, self.cos, self.sin, offset=offset)
            ref = _ref_rope_apply_torch(torch.tensor(x), self.cos, self.sin, offset=offset)
            np.testing.assert_allclose(out, torch_to_np(ref), atol=1e-5, err_msg=f"offset={offset}")

    def test_norm_is_preserved(self):
        out = attn.rope_apply(self.x, self.cos, self.sin)
        np.testing.assert_allclose(
            np.linalg.norm(out, axis=-1), np.linalg.norm(self.x, axis=-1), atol=1e-4
        )

    def test_dot_product_depends_only_on_relative_distance(self):
        """The defining property: <rope(q,m), rope(k,n)> is a function of m-n."""
        set_seed(37)
        q = np.random.randn(1, 1, 1, self.head_dim).astype(np.float32)
        k = np.random.randn(1, 1, 1, self.head_dim).astype(np.float32)

        def dot(m, n):
            qm = attn.rope_apply(q, self.cos, self.sin, offset=m)
            kn = attn.rope_apply(k, self.cos, self.sin, offset=n)
            return float((qm * kn).sum())

        self.assertAlmostEqual(dot(5, 3), dot(9, 7), places=4)
        self.assertAlmostEqual(dot(2, 6), dot(8, 12), places=4)
        self.assertNotAlmostEqual(dot(5, 3), dot(5, 4), places=4)

    def test_backward(self):
        dout = np.random.randn(*self.x.shape).astype(np.float32)
        dx = attn.rope_backward(dout, self.cos, self.sin)

        x_th = torch.tensor(self.x, requires_grad=True)
        _ref_rope_apply_torch(x_th, self.cos, self.sin).backward(torch.tensor(dout))
        np.testing.assert_allclose(dx, torch_to_np(x_th.grad), atol=1e-5)

    def test_backward_with_offset(self):
        x = self.x[:, :, :2, :]
        dout = np.random.randn(*x.shape).astype(np.float32)
        dx = attn.rope_backward(dout, self.cos, self.sin, offset=4)

        x_th = torch.tensor(x, requires_grad=True)
        _ref_rope_apply_torch(x_th, self.cos, self.sin, offset=4).backward(torch.tensor(dout))
        np.testing.assert_allclose(dx, torch_to_np(x_th.grad), atol=1e-5)


class TestALiBi(unittest.TestCase):
    def test_slopes_power_of_two(self):
        slopes = attn.alibi_slopes(8)
        self.assertEqual(slopes.shape, (8,))
        np.testing.assert_allclose(
            slopes, [2.0 ** -(i + 1) for i in range(8)], atol=1e-6
        )

    def test_slopes_are_decreasing_and_positive(self):
        for n in (1, 2, 4, 16):
            s = attn.alibi_slopes(n)
            self.assertTrue(np.all(s > 0), f"n_heads={n}")
            self.assertTrue(np.all(np.diff(s) < 0) or n == 1, f"n_heads={n}")

    def test_bias_shape_and_sign(self):
        bias = attn.alibi_bias(4, 5, 5)
        self.assertEqual(bias.shape, (1, 4, 5, 5))
        self.assertTrue(np.all(bias <= 1e-7))
        # Zero penalty when query and key are at the same position
        for h in range(4):
            np.testing.assert_allclose(np.diag(bias[0, h]), 0.0, atol=1e-6)

    def test_bias_is_linear_in_distance(self):
        slopes = attn.alibi_slopes(4)
        bias = attn.alibi_bias(4, 6, 6)
        for h in range(4):
            for i in range(6):
                for j in range(6):
                    self.assertAlmostEqual(
                        float(bias[0, h, i, j]), -float(slopes[h]) * abs(i - j), places=5
                    )

    def test_bias_with_offset(self):
        """During decoding the single query sits at absolute position `offset`."""
        slopes = attn.alibi_slopes(2)
        bias = attn.alibi_bias(2, 1, 5, offset=4)
        self.assertEqual(bias.shape, (1, 2, 1, 5))
        for h in range(2):
            for j in range(5):
                self.assertAlmostEqual(
                    float(bias[0, h, 0, j]), -float(slopes[h]) * abs(4 - j), places=5
                )


# ---------------------------------------------------------------------------
# Layer 7 — Training machinery
# ---------------------------------------------------------------------------

class TestCrossEntropy(unittest.TestCase):
    def setUp(self):
        set_seed(40)
        self.N, self.vocab = 12, 7
        self.logits = np.random.randn(self.N, self.vocab).astype(np.float32) * 3.0
        self.targets = np.random.randint(0, self.vocab, size=(self.N,)).astype(np.int64)

    def _compare(self, logits, targets, ignore_index=-100, label_smoothing=0.0, flat_dims=1):
        loss, cache = attn.cross_entropy_forward(
            logits, targets, ignore_index=ignore_index, label_smoothing=label_smoothing
        )
        dlogits = attn.cross_entropy_backward(cache)

        l_th = torch.tensor(logits, requires_grad=True)
        t_th = torch.tensor(targets)
        if flat_dims == 2:  # (batch, seq, vocab) -> (batch, vocab, seq)
            loss_th = F.cross_entropy(
                l_th.permute(0, 2, 1), t_th,
                ignore_index=ignore_index, label_smoothing=label_smoothing,
            )
        else:
            loss_th = F.cross_entropy(
                l_th, t_th, ignore_index=ignore_index, label_smoothing=label_smoothing
            )
        loss_th.backward()

        self.assertAlmostEqual(float(loss), float(loss_th.item()), places=5)
        self.assertEqual(dlogits.shape, logits.shape)
        np.testing.assert_allclose(dlogits, torch_to_np(l_th.grad), atol=1e-6)

    def test_basic(self):
        self._compare(self.logits, self.targets)

    def test_three_dimensional_input(self):
        set_seed(41)
        logits = np.random.randn(3, 5, self.vocab).astype(np.float32) * 2.0
        targets = np.random.randint(0, self.vocab, size=(3, 5)).astype(np.int64)
        self._compare(logits, targets, flat_dims=2)

    def test_ignore_index(self):
        targets = self.targets.copy()
        targets[[1, 4, 9]] = -100
        self._compare(self.logits, targets, ignore_index=-100)

    def test_ignore_index_zeroes_gradient(self):
        targets = self.targets.copy()
        targets[2] = -100
        _, cache = attn.cross_entropy_forward(self.logits, targets)
        dlogits = attn.cross_entropy_backward(cache)
        np.testing.assert_allclose(dlogits[2], 0.0, atol=1e-9)

    def test_label_smoothing(self):
        self._compare(self.logits, self.targets, label_smoothing=0.1)

    def test_label_smoothing_with_ignore_index(self):
        targets = self.targets.copy()
        targets[[0, 3]] = -100
        self._compare(self.logits, targets, ignore_index=-100, label_smoothing=0.2)

    def test_numerical_stability(self):
        logits = np.array([[0.0, 500.0, -500.0]], dtype=np.float32)
        targets = np.array([1], dtype=np.int64)
        loss, cache = attn.cross_entropy_forward(logits, targets)
        self.assertFalse(np.isnan(loss))
        self.assertFalse(np.any(np.isnan(attn.cross_entropy_backward(cache))))

    def test_perfect_prediction_gives_near_zero_loss(self):
        logits = np.full((1, 4), -50.0, dtype=np.float32)
        logits[0, 2] = 50.0
        loss, _ = attn.cross_entropy_forward(logits, np.array([2], dtype=np.int64))
        self.assertLess(float(loss), 1e-5)


class TestClipGradNorm(unittest.TestCase):
    def _torch_ref(self, arrays, max_norm, norm_type=2.0):
        params = [torch.nn.Parameter(torch.tensor(a.copy())) for a in arrays]
        for p, a in zip(params, arrays):
            p.grad = torch.tensor(a.copy())
        total = torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=norm_type)
        return float(total), [torch_to_np(p.grad) for p in params]

    def test_clips_when_over_threshold(self):
        set_seed(42)
        grads = [np.random.randn(4, 5).astype(np.float32) * 5, np.random.randn(7).astype(np.float32) * 5]
        ref_norm, ref_grads = self._torch_ref(grads, 1.0)

        mine = [g.copy() for g in grads]
        got_norm = attn.clip_grad_norm(mine, 1.0)

        self.assertAlmostEqual(got_norm, ref_norm, places=4)
        for a, b in zip(mine, ref_grads):
            np.testing.assert_allclose(a, b, atol=1e-5)
        # After clipping the global norm is exactly max_norm
        flat = np.concatenate([g.ravel() for g in mine])
        self.assertAlmostEqual(float(np.linalg.norm(flat)), 1.0, places=4)

    def test_no_op_when_under_threshold(self):
        set_seed(43)
        grads = [np.random.randn(3, 3).astype(np.float32) * 0.01]
        before = grads[0].copy()
        norm = attn.clip_grad_norm(grads, 100.0)
        np.testing.assert_array_equal(grads[0], before)
        self.assertLess(norm, 100.0)

    def test_is_global_not_per_tensor(self):
        """Two tensors of norm 3 and 4 must both be scaled by 1/5, not to 1 each."""
        a = np.array([3.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 4.0], dtype=np.float32)
        grads = [a.copy(), b.copy()]
        norm = attn.clip_grad_norm(grads, 1.0)
        self.assertAlmostEqual(norm, 5.0, places=5)
        np.testing.assert_allclose(grads[0], a / 5.0, atol=1e-6)
        np.testing.assert_allclose(grads[1], b / 5.0, atol=1e-6)

    def test_inf_norm(self):
        set_seed(44)
        grads = [np.random.randn(4).astype(np.float32) * 10]
        ref_norm, ref_grads = self._torch_ref(grads, 1.0, norm_type=float("inf"))
        mine = [g.copy() for g in grads]
        got = attn.clip_grad_norm(mine, 1.0, norm_type=np.inf)
        self.assertAlmostEqual(got, ref_norm, places=4)
        np.testing.assert_allclose(mine[0], ref_grads[0], atol=1e-5)


class TestLRSchedule(unittest.TestCase):
    def test_warmup_is_linear_and_nonzero_at_step_zero(self):
        lrs = [attn.lr_cosine_with_warmup(s, 1.0, 10, 100) for s in range(10)]
        self.assertGreater(lrs[0], 0.0)
        diffs = np.diff(lrs)
        np.testing.assert_allclose(diffs, diffs[0], atol=1e-9)

    def test_peak_at_end_of_warmup(self):
        lr = attn.lr_cosine_with_warmup(10, 1.0, 10, 100)
        self.assertAlmostEqual(lr, 1.0, places=6)

    def test_monotone_decay_after_warmup(self):
        lrs = [attn.lr_cosine_with_warmup(s, 1.0, 10, 100) for s in range(10, 101)]
        self.assertTrue(np.all(np.diff(lrs) <= 1e-9))

    def test_floor_at_and_after_total(self):
        self.assertAlmostEqual(attn.lr_cosine_with_warmup(100, 1.0, 10, 100, min_lr=0.1), 0.1, places=6)
        self.assertAlmostEqual(attn.lr_cosine_with_warmup(500, 1.0, 10, 100, min_lr=0.1), 0.1, places=6)

    def test_midpoint_of_cosine(self):
        """Halfway through decay, a cosine schedule sits at the midpoint of the range."""
        lr = attn.lr_cosine_with_warmup(55, 1.0, 10, 100, min_lr=0.0)
        self.assertAlmostEqual(lr, 0.5, places=5)

    def test_no_warmup(self):
        lr = attn.lr_cosine_with_warmup(0, 1.0, 0, 100)
        self.assertAlmostEqual(lr, 1.0, places=6)


class _OptimizerFixture:
    """Runs an optimiser for several steps against its PyTorch counterpart."""

    def _run(self, make_mine, make_torch, n_steps=5, atol=1e-6):
        set_seed(45)
        shapes = [(4, 3), (5,)]
        init = [np.random.randn(*s).astype(np.float32) for s in shapes]
        grad_seq = [
            [np.random.randn(*s).astype(np.float32) for s in shapes] for _ in range(n_steps)
        ]

        mine_params = [p.copy() for p in init]
        opt_mine = make_mine(mine_params)

        th_params = [torch.nn.Parameter(torch.tensor(p.copy())) for p in init]
        opt_th = make_torch(th_params)

        for step, grads in enumerate(grad_seq):
            opt_mine.step([g.copy() for g in grads])
            for p, g in zip(th_params, grads):
                p.grad = torch.tensor(g.copy())
            opt_th.step()
            opt_th.zero_grad(set_to_none=False)

            for i, (a, b) in enumerate(zip(mine_params, th_params)):
                np.testing.assert_allclose(
                    a, torch_to_np(b), atol=atol,
                    err_msg=f"param {i} diverged at step {step}",
                )


class TestSGD(_OptimizerFixture, unittest.TestCase):
    def test_vanilla(self):
        self._run(
            lambda p: attn.SGD(p, lr=0.1),
            lambda p: torch.optim.SGD(p, lr=0.1),
        )

    def test_momentum(self):
        self._run(
            lambda p: attn.SGD(p, lr=0.1, momentum=0.9),
            lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9),
        )

    def test_weight_decay(self):
        self._run(
            lambda p: attn.SGD(p, lr=0.1, momentum=0.9, weight_decay=0.05),
            lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, weight_decay=0.05),
        )

    def test_nesterov(self):
        self._run(
            lambda p: attn.SGD(p, lr=0.1, momentum=0.9, nesterov=True),
            lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, nesterov=True),
        )

    def test_updates_in_place(self):
        params = [np.ones((2, 2), dtype=np.float32)]
        holder = params[0]
        attn.SGD(params, lr=1.0).step([np.ones((2, 2), dtype=np.float32)])
        np.testing.assert_allclose(holder, 0.0, atol=1e-6)


class TestAdamW(_OptimizerFixture, unittest.TestCase):
    def test_default_hyperparameters(self):
        self._run(
            lambda p: attn.AdamW(p, lr=1e-2),
            lambda p: torch.optim.AdamW(p, lr=1e-2),
            atol=1e-6,
        )

    def test_no_weight_decay_reduces_to_adam(self):
        self._run(
            lambda p: attn.AdamW(p, lr=1e-2, weight_decay=0.0),
            lambda p: torch.optim.Adam(p, lr=1e-2),
            atol=1e-6,
        )

    def test_custom_betas_and_decay(self):
        self._run(
            lambda p: attn.AdamW(p, lr=5e-3, betas=(0.8, 0.95), eps=1e-8, weight_decay=0.1),
            lambda p: torch.optim.AdamW(p, lr=5e-3, betas=(0.8, 0.95), eps=1e-8, weight_decay=0.1),
            atol=1e-6,
        )

    def test_first_step_magnitude_is_roughly_lr(self):
        """Bias correction makes the very first update ~lr in magnitude, whatever the grad scale."""
        params = [np.zeros((100,), dtype=np.float32)]
        opt = attn.AdamW(params, lr=1e-3, weight_decay=0.0)
        opt.step([np.full((100,), 1000.0, dtype=np.float32)])
        np.testing.assert_allclose(np.abs(params[0]), 1e-3, rtol=1e-3)

    def test_step_counter_advances(self):
        params = [np.zeros((3,), dtype=np.float32)]
        opt = attn.AdamW(params, lr=1e-3)
        for expected in (1, 2, 3):
            opt.step([np.ones((3,), dtype=np.float32)])
            self.assertEqual(opt.t, expected)


# ---------------------------------------------------------------------------
# Layer 8 — Inference: KV cache, incremental decoding, sampling
# ---------------------------------------------------------------------------

class TestKVCache(unittest.TestCase):
    def setUp(self):
        self.batch, self.n_kv, self.max_len, self.dim = 2, 3, 8, 4
        self.cache = attn.KVCache(self.batch, self.n_kv, self.max_len, self.dim)

    def test_prefill_then_decode_lengths(self):
        k = np.random.randn(self.batch, self.n_kv, 5, self.dim).astype(np.float32)
        K, V = self.cache.append(k, k * 2)
        self.assertEqual(self.cache.length, 5)
        self.assertEqual(K.shape, (self.batch, self.n_kv, 5, self.dim))
        np.testing.assert_allclose(K, k, atol=1e-6)
        np.testing.assert_allclose(V, k * 2, atol=1e-6)

        step = np.random.randn(self.batch, self.n_kv, 1, self.dim).astype(np.float32)
        K, V = self.cache.append(step, step * 2)
        self.assertEqual(self.cache.length, 6)
        self.assertEqual(K.shape, (self.batch, self.n_kv, 6, self.dim))
        np.testing.assert_allclose(K[:, :, :5], k, atol=1e-6)
        np.testing.assert_allclose(K[:, :, 5:], step, atol=1e-6)

    def test_token_by_token_matches_bulk(self):
        k = np.random.randn(self.batch, self.n_kv, 6, self.dim).astype(np.float32)
        for t in range(6):
            K, V = self.cache.append(k[:, :, t:t + 1], k[:, :, t:t + 1] * 3)
        np.testing.assert_allclose(K, k, atol=1e-6)
        np.testing.assert_allclose(V, k * 3, atol=1e-6)

    def test_never_returns_padded_zeros(self):
        k = np.ones((self.batch, self.n_kv, 2, self.dim), dtype=np.float32)
        K, _ = self.cache.append(k, k)
        self.assertEqual(K.shape[2], 2)
        self.assertTrue(np.all(K != 0.0))

    def test_overflow_is_rejected(self):
        k = np.zeros((self.batch, self.n_kv, self.max_len + 1, self.dim), dtype=np.float32)
        with self.assertRaises((ValueError, AssertionError, IndexError)):
            self.cache.append(k, k)

    def test_reset(self):
        k = np.random.randn(self.batch, self.n_kv, 3, self.dim).astype(np.float32)
        self.cache.append(k, k)
        self.cache.reset()
        self.assertEqual(self.cache.length, 0)
        k2 = np.random.randn(self.batch, self.n_kv, 1, self.dim).astype(np.float32)
        K, _ = self.cache.append(k2, k2)
        self.assertEqual(K.shape[2], 1)
        np.testing.assert_allclose(K, k2, atol=1e-6)


class TestIncrementalDecoding(_MHAFixture, unittest.TestCase):
    """The single most valuable test in this file: cached decoding must be
    bit-comparable to a full causal forward pass over the same tokens."""

    def test_matches_full_causal_forward(self):
        x = np.random.randn(self.batch, self.seq_q, self.d_model).astype(np.float32)
        causal = attn.create_causal_mask(self.seq_q)
        full = self.np_mha.forward(x, x, x, mask=causal, training=False)

        cache = attn.KVCache(self.batch, self.n_heads, self.seq_q, self.head_dim)
        steps = [
            self.np_mha.forward_incremental(x[:, t:t + 1, :], cache)
            for t in range(self.seq_q)
        ]
        got = np.concatenate(steps, axis=1)

        self.assertEqual(got.shape, full.shape)
        np.testing.assert_allclose(got, full, atol=1e-5)

    def test_cache_grows_one_per_step(self):
        x = np.random.randn(self.batch, self.seq_q, self.d_model).astype(np.float32)
        cache = attn.KVCache(self.batch, self.n_heads, self.seq_q, self.head_dim)
        for t in range(self.seq_q):
            self.np_mha.forward_incremental(x[:, t:t + 1, :], cache)
            self.assertEqual(cache.length, t + 1)

    def test_step_output_shape(self):
        x = np.random.randn(self.batch, 1, self.d_model).astype(np.float32)
        cache = attn.KVCache(self.batch, self.n_heads, self.seq_q, self.head_dim)
        out = self.np_mha.forward_incremental(x, cache)
        self.assertEqual(out.shape, (self.batch, 1, self.d_model))


class TestSampling(unittest.TestCase):
    def setUp(self):
        # A deliberately skewed distribution: probs = [0.5, 0.3, 0.15, 0.05]
        self.probs = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float64)
        self.logits = np.log(self.probs).astype(np.float32)[None, :]

    def test_greedy_select(self):
        logits = np.random.randn(2, 3, 5).astype(np.float32)
        idx = attn.greedy_select(logits)
        self.assertEqual(idx.shape, (2, 3))
        np.testing.assert_array_equal(idx, np.argmax(logits, axis=-1))

    def test_temperature_scales_logits(self):
        out = attn.apply_temperature(self.logits, 2.0)
        np.testing.assert_allclose(out, self.logits / 2.0, atol=1e-6)

    def test_low_temperature_sharpens(self):
        sharp = attn.softmax(attn.apply_temperature(self.logits, 0.1), axis=-1)
        flat = attn.softmax(attn.apply_temperature(self.logits, 10.0), axis=-1)
        self.assertGreater(float(sharp.max()), 0.99)
        self.assertLess(float(flat.max()), 0.4)
        # Ordering never changes, only the concentration
        np.testing.assert_array_equal(np.argsort(-sharp[0]), np.argsort(-flat[0]))

    def test_top_k_keeps_exactly_k(self):
        out = attn.top_k_filter(self.logits, 2)
        self.assertEqual(int(np.isfinite(out).sum()), 2)
        np.testing.assert_array_equal(np.isfinite(out[0]), [True, True, False, False])

    def test_top_k_does_not_mutate_input(self):
        before = self.logits.copy()
        attn.top_k_filter(self.logits, 1)
        np.testing.assert_array_equal(self.logits, before)

    def test_top_k_degenerate(self):
        np.testing.assert_array_equal(attn.top_k_filter(self.logits, 0), self.logits)
        np.testing.assert_array_equal(attn.top_k_filter(self.logits, 99), self.logits)

    def test_top_k_batched_rows_independent(self):
        logits = np.array([[3.0, 1.0, 2.0], [1.0, 5.0, 4.0]], dtype=np.float32)
        out = attn.top_k_filter(logits, 2)
        np.testing.assert_array_equal(np.isfinite(out), [[True, False, True], [False, True, True]])

    def test_top_p_keeps_crossing_token(self):
        """p=0.55: cumulative is 0.5 then 0.8, so the second token must survive."""
        out = attn.top_p_filter(self.logits, 0.55)
        np.testing.assert_array_equal(np.isfinite(out[0]), [True, True, False, False])

    def test_top_p_below_first_token(self):
        """p=0.45 < 0.5, so the single most likely token already suffices."""
        out = attn.top_p_filter(self.logits, 0.45)
        np.testing.assert_array_equal(np.isfinite(out[0]), [True, False, False, False])

    def test_top_p_always_keeps_at_least_one(self):
        out = attn.top_p_filter(self.logits, 1e-6)
        self.assertGreaterEqual(int(np.isfinite(out).sum()), 1)

    def test_top_p_one_is_identity(self):
        np.testing.assert_array_equal(attn.top_p_filter(self.logits, 1.0), self.logits)

    def test_top_p_respects_ordering_not_position(self):
        logits = np.log(np.array([[0.05, 0.5, 0.15, 0.3]], dtype=np.float64)).astype(np.float32)
        out = attn.top_p_filter(logits, 0.55)
        np.testing.assert_array_equal(np.isfinite(out[0]), [False, True, False, True])

    def test_sample_shape_and_dtype(self):
        rng = np.random.default_rng(0)
        idx = attn.sample_from_logits(np.tile(self.logits, (5, 1)), rng)
        self.assertEqual(idx.shape, (5,))
        self.assertTrue(np.issubdtype(idx.dtype, np.integer))
        self.assertTrue(np.all((idx >= 0) & (idx < 4)))

    def test_sample_is_reproducible(self):
        logits = np.tile(self.logits, (20, 1))
        a = attn.sample_from_logits(logits, np.random.default_rng(7))
        b = attn.sample_from_logits(logits, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)

    def test_sample_matches_distribution(self):
        rng = np.random.default_rng(1)
        draws = attn.sample_from_logits(np.tile(self.logits, (20000, 1)), rng)
        freq = np.bincount(draws, minlength=4) / 20000.0
        np.testing.assert_allclose(freq, self.probs, atol=0.02)

    def test_masked_tokens_are_never_sampled(self):
        logits = np.tile(self.logits, (3000, 1))
        filtered = attn.top_k_filter(logits, 2)
        draws = attn.sample_from_logits(filtered, np.random.default_rng(2))
        self.assertTrue(np.all(draws < 2))

    def test_top_k_one_then_sample_equals_greedy(self):
        logits = np.random.randn(50, 9).astype(np.float32)
        draws = attn.sample_from_logits(attn.top_k_filter(logits, 1), np.random.default_rng(3))
        np.testing.assert_array_equal(draws, attn.greedy_select(logits))


# ---------------------------------------------------------------------------
# Layer 9 — FlashAttention
# ---------------------------------------------------------------------------

class TestFlashAttention(unittest.TestCase):
    def setUp(self):
        set_seed(50)
        self.batch, self.heads, self.seq, self.d = 2, 3, 20, 8
        self.Q = np.random.randn(self.batch, self.heads, self.seq, self.d).astype(np.float32)
        self.K = np.random.randn(self.batch, self.heads, self.seq, self.d).astype(np.float32)
        self.V = np.random.randn(self.batch, self.heads, self.seq, self.d).astype(np.float32)

    def _torch_out(self, causal):
        Q = torch.tensor(self.Q, requires_grad=True)
        K = torch.tensor(self.K, requires_grad=True)
        V = torch.tensor(self.V, requires_grad=True)
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
        return out, Q, K, V

    def test_forward_non_causal(self):
        out, _ = attn.flash_attention_forward(self.Q, self.K, self.V, block_q=8, block_k=8)
        ref, *_ = self._torch_out(causal=False)
        self.assertEqual(out.shape, (self.batch, self.heads, self.seq, self.d))
        np.testing.assert_allclose(out, torch_to_np(ref), atol=1e-5)

    def test_forward_causal(self):
        out, _ = attn.flash_attention_forward(self.Q, self.K, self.V, block_q=8, block_k=8, causal=True)
        ref, *_ = self._torch_out(causal=True)
        np.testing.assert_allclose(out, torch_to_np(ref), atol=1e-5)

    def test_forward_invariant_to_block_size(self):
        """Tiling is an implementation detail; the answer must not depend on it."""
        ref, *_ = self._torch_out(causal=False)
        for bq, bk in [(1, 1), (3, 7), (20, 20), (64, 64)]:
            out, _ = attn.flash_attention_forward(self.Q, self.K, self.V, block_q=bq, block_k=bk)
            np.testing.assert_allclose(
                out, torch_to_np(ref), atol=1e-5, err_msg=f"block_q={bq}, block_k={bk}"
            )

    def test_cache_stores_logsumexp(self):
        _, cache = attn.flash_attention_forward(self.Q, self.K, self.V, block_q=8, block_k=8)
        self.assertIn("L", cache, "cache must expose the per-row log-sum-exp as 'L'")
        L = cache["L"]
        self.assertEqual(L.shape, (self.batch, self.heads, self.seq, 1))

        scores = torch.tensor(self.Q) @ torch.tensor(self.K).transpose(-1, -2) / np.sqrt(self.d)
        ref = torch.logsumexp(scores, dim=-1, keepdim=True)
        np.testing.assert_allclose(L, torch_to_np(ref), atol=1e-4)

    def test_matches_layer2_implementation(self):
        """Cross-check against your own non-tiled attention, not just PyTorch."""
        flash, _ = attn.flash_attention_forward(self.Q, self.K, self.V, block_q=8, block_k=8)
        naive, _ = attn.scaled_dot_product_attention_forward(self.Q, self.K, self.V)
        np.testing.assert_allclose(flash, naive, atol=1e-5)

    def test_backward_non_causal(self):
        out, cache = attn.flash_attention_forward(self.Q, self.K, self.V, block_q=8, block_k=8)
        dout = np.random.randn(*out.shape).astype(np.float32)
        dQ, dK, dV = attn.flash_attention_backward(dout, cache, block_q=8, block_k=8)

        ref, Q, K, V = self._torch_out(causal=False)
        ref.backward(torch.tensor(dout))

        np.testing.assert_allclose(dQ, torch_to_np(Q.grad), atol=1e-5)
        np.testing.assert_allclose(dK, torch_to_np(K.grad), atol=1e-5)
        np.testing.assert_allclose(dV, torch_to_np(V.grad), atol=1e-5)

    def test_backward_causal(self):
        out, cache = attn.flash_attention_forward(
            self.Q, self.K, self.V, block_q=8, block_k=8, causal=True
        )
        dout = np.random.randn(*out.shape).astype(np.float32)
        dQ, dK, dV = attn.flash_attention_backward(dout, cache, block_q=8, block_k=8)

        ref, Q, K, V = self._torch_out(causal=True)
        ref.backward(torch.tensor(dout))

        np.testing.assert_allclose(dQ, torch_to_np(Q.grad), atol=1e-5)
        np.testing.assert_allclose(dK, torch_to_np(K.grad), atol=1e-5)
        np.testing.assert_allclose(dV, torch_to_np(V.grad), atol=1e-5)

    def test_backward_invariant_to_block_size(self):
        out, cache = attn.flash_attention_forward(self.Q, self.K, self.V, block_q=4, block_k=4)
        dout = np.random.randn(*out.shape).astype(np.float32)
        ref = attn.flash_attention_backward(dout, cache, block_q=4, block_k=4)
        got = attn.flash_attention_backward(dout, cache, block_q=6, block_k=9)
        for a, b in zip(got, ref):
            np.testing.assert_allclose(a, b, atol=1e-5)

    def test_long_sequence_stays_within_memory_budget(self):
        """The whole point: peak memory must not scale with seq_q * seq_k.

        A 4096-long score matrix would be 64 MB per (batch, head); if the
        implementation materialises it, this test balloons in memory. Kept
        moderate so a mistake is slow-but-survivable rather than an OOM crash.
        """
        set_seed(51)
        seq = 512
        Q = np.random.randn(1, 1, seq, 16).astype(np.float32)
        K = np.random.randn(1, 1, seq, 16).astype(np.float32)
        V = np.random.randn(1, 1, seq, 16).astype(np.float32)
        out, cache = attn.flash_attention_forward(Q, K, V, block_q=64, block_k=64, causal=True)
        ref = F.scaled_dot_product_attention(
            torch.tensor(Q), torch.tensor(K), torch.tensor(V), is_causal=True
        )
        np.testing.assert_allclose(out, torch_to_np(ref), atol=1e-4)
        self.assertEqual(cache["L"].shape, (1, 1, seq, 1))


# ---------------------------------------------------------------------------
# Layer 10 — Quantisation
# ---------------------------------------------------------------------------

class TestQuantization(unittest.TestCase):
    def test_per_tensor_dtype_and_range(self):
        set_seed(60)
        x = np.random.randn(16, 32).astype(np.float32) * 4.0
        q, scale = attn.quantize_symmetric_int8(x)
        self.assertEqual(q.dtype, np.int8)
        self.assertEqual(q.shape, x.shape)
        self.assertLessEqual(int(np.abs(q).max()), 127)
        self.assertGreaterEqual(int(np.abs(q).max()), 120)  # absmax must actually saturate

    def test_round_trip_error_is_bounded_by_half_a_step(self):
        set_seed(61)
        x = np.random.randn(64, 64).astype(np.float32) * 3.0
        q, scale = attn.quantize_symmetric_int8(x)
        x_hat = attn.dequantize_int8(q, scale)
        self.assertEqual(x_hat.shape, x.shape)
        self.assertLessEqual(float(np.abs(x - x_hat).max()), float(np.max(scale)) * 0.5 + 1e-6)

    def test_extremes_map_to_the_endpoints(self):
        x = np.array([-2.0, 0.0, 1.0, 2.0], dtype=np.float32)
        q, scale = attn.quantize_symmetric_int8(x)
        self.assertEqual(int(q[3]), 127)
        self.assertEqual(int(q[0]), -127)
        self.assertEqual(int(q[1]), 0)

    def test_all_zeros_does_not_produce_nan(self):
        x = np.zeros((4, 4), dtype=np.float32)
        q, scale = attn.quantize_symmetric_int8(x)
        self.assertFalse(np.any(np.isnan(attn.dequantize_int8(q, scale))))
        np.testing.assert_array_equal(q, 0)

    def test_per_channel_scales(self):
        """Rows with wildly different magnitudes are what per-tensor scaling ruins."""
        x = np.stack([
            np.linspace(-1.0, 1.0, 32),
            np.linspace(-1000.0, 1000.0, 32),
        ]).astype(np.float32)

        q_pc, s_pc = attn.quantize_symmetric_int8(x, axis=0)
        self.assertEqual(np.asarray(s_pc).size, 2)
        err_pc = np.abs(x - attn.dequantize_int8(q_pc, s_pc))

        q_pt, s_pt = attn.quantize_symmetric_int8(x, axis=None)
        err_pt = np.abs(x - attn.dequantize_int8(q_pt, s_pt))

        # The small-magnitude row is destroyed by a single global scale
        self.assertLess(err_pc[0].max(), err_pt[0].max() / 10.0)

    def test_matmul_accumulates_in_int32(self):
        """K=4096 with saturated int8 inputs overflows int16 but fits int32."""
        K = 4096
        a_q = np.full((1, K), 127, dtype=np.int8)
        b_q = np.full((K, 1), 127, dtype=np.int8)
        one = np.float32(1.0)
        out = attn.quantized_matmul_int8(a_q, one, b_q, one)
        self.assertEqual(out.dtype, np.float32)
        self.assertAlmostEqual(float(out[0, 0]), 127.0 * 127.0 * K, delta=1.0)

    def test_matmul_approximates_float_matmul(self):
        set_seed(62)
        A = np.random.randn(32, 64).astype(np.float32)
        B = np.random.randn(64, 16).astype(np.float32)

        a_q, a_s = attn.quantize_symmetric_int8(A)
        b_q, b_s = attn.quantize_symmetric_int8(B)
        got = attn.quantized_matmul_int8(a_q, a_s, b_q, b_s)

        ref = A @ B
        self.assertEqual(got.shape, ref.shape)
        rel = np.abs(got - ref).max() / np.abs(ref).max()
        self.assertLess(float(rel), 0.05)

    def test_matmul_with_per_channel_scales(self):
        set_seed(63)
        A = np.random.randn(8, 16).astype(np.float32)
        B = np.random.randn(16, 4).astype(np.float32)

        a_q, a_s = attn.quantize_symmetric_int8(A)
        b_q, b_s = attn.quantize_symmetric_int8(B, axis=1)  # per output channel
        got = attn.quantized_matmul_int8(a_q, a_s, b_q, b_s)

        ref = A @ B
        rel = np.abs(got - ref).max() / np.abs(ref).max()
        self.assertLess(float(rel), 0.05)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run everything:      python -m unittest test_attention -v
    # Run one topic:       python -m unittest test_attention.TestRoPE -v
    unittest.main(verbosity=2)