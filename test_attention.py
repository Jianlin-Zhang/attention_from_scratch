"""
Test Suite for attention.py
===========================
Run with the ``openvla`` conda environment (PyTorch 2.2.0, NumPy 1.26.4):

    conda activate openvla
    python -m pytest test_attention.py -v

or simply:

    python test_attention.py

Each test compares the pure-NumPy implementation against PyTorch as ground truth.
Work through the tests layer by layer — implement the corresponding TODO in
attention.py, then run its test to validate.
"""

import sys
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
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run with:  python test_attention.py
    unittest.main(verbosity=2)