"""
Attention Mechanisms from Scratch — Pure NumPy Implementation
=============================================================
Layered from low-level utilities up to a full Transformer attention block.

Each component has TODO markers for you to fill in.  Run the tests
alongside to validate your implementation step by step:

    conda activate openvla
    python test_attention.py

Conventions
-----------
* Batch-first: all tensors are ``(batch, seq_len, features)``.
* Masks are **additive**: ``0`` = allowed, ``-inf`` = masked (added before softmax).
* Forward returns a ``cache`` dict of intermediates; backward consumes it.
"""

import numpy as np


# =============================================================================
# Layer 0 — Utilities
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax.

    Parameters
    ----------
    x : shape (..., D)
    axis : axis to normalise over

    Returns
    -------
    probs : same shape as x, sums to 1 along *axis*
    """
    # TODO: subtract max, exp, then divide by sum
    # Hint: use np.max(..., keepdims=True), np.exp, np.sum(..., keepdims=True)
    raise NotImplementedError


def softmax_backward(dout: np.ndarray, probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Gradient of softmax.

    Given  y = softmax(x)  and  dL/dy = dout, returns  dL/dx.

    Formula:  dL/dx_i = y_i * (dout_i - Σ_j dout_j * y_j)
    """
    # TODO: implement the softmax gradient formula above
    # Hint: compute sum(dout * probs) along axis, then probs * (dout - sum_term)
    raise NotImplementedError


def create_causal_mask(seq_len: int) -> np.ndarray:
    """Create an upper-triangular mask for autoregressive / causal attention.

    Parameters
    ----------
    seq_len : query (and key) sequence length

    Returns
    -------
    mask : shape ``(seq_len, seq_len)``, dtype float32
           Lower triangle (incl. diagonal) = 0, upper triangle = -inf.
    """
    # TODO: use np.triu to create the mask
    # Hint: np.triu(np.ones(...), k=1) → upper triangle is 1, then convert 1 → -inf, 0 → 0
    raise NotImplementedError


def create_padding_mask(valid_lens: np.ndarray, max_len: int) -> np.ndarray:
    """Create a mask that hides padding positions.

    Parameters
    ----------
    valid_lens : shape ``(batch,)`` — number of valid (non-padding) tokens per sequence.
    max_len   : maximum sequence length

    Returns
    -------
    mask : shape ``(batch, 1, 1, max_len)``, dtype float32
           Positions < valid_len → 0,  positions >= valid_len → -inf.
    """
    # TODO: create a boolean mask from valid_lens and convert to 0/-inf
    # Hint: create (batch, 1, 1, max_len) zeros, then set mask[:,:,:,valid_lens[i]:] = -inf
    raise NotImplementedError


def dropout_forward(x: np.ndarray, p: float, training: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Inverted dropout: scales kept neurons by 1/(1-p) during training.

    Returns (output, mask).  Mask is ``None`` when not training.
    """
    # TODO: if not training or p <= 0, return (x, None)
    # Otherwise: generate random mask, apply mask, scale by 1/(1-p)
    # Hint: mask = (np.random.rand(*x.shape) > p).astype(np.float32)
    raise NotImplementedError


def dropout_backward(dout: np.ndarray, mask: np.ndarray | None, p: float) -> np.ndarray:
    """Backward pass for inverted dropout."""
    # TODO: if mask is None, return dout. Otherwise scale by mask / (1-p)
    raise NotImplementedError


# =============================================================================
# Layer 1 — Linear
# =============================================================================

class Linear:
    """A simple linear (fully-connected) layer:  y = x @ W + b.

    Parameters
    ----------
    in_features : int
    out_features : int
    """

    def __init__(self, in_features: int, out_features: int):
        # He initialization: sqrt(2 / fan_in)
        self.W: np.ndarray = np.random.randn(in_features, out_features).astype(np.float32) * np.sqrt(2.0 / in_features)
        self.b: np.ndarray = np.zeros(out_features, dtype=np.float32)
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None
        self._cache: dict | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.

        Parameters
        ----------
        x : shape ``(..., in_features)``

        Returns
        -------
        y : shape ``(..., out_features)``
        """
        # TODO: y = x @ W + b.  Store x in self._cache for backward.
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass.

        Parameters
        ----------
        dout : gradient w.r.t. output, same shape as y

        Returns
        -------
        dx : gradient w.r.t. input, same shape as x
        """
        # TODO: compute dW, db, dx
        # dW = x_flat^T @ dout_flat  (reshape ... to (-1, in_features) first)
        # db = sum(dout_flat, axis=0)
        # dx = dout @ W^T
        raise NotImplementedError


# =============================================================================
# Layer 2 — Scaled Dot-Product Attention (single head)
# =============================================================================

def scaled_dot_product_attention_forward(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Single-head scaled dot-product attention — forward.

    Parameters
    ----------
    Q : shape ``(batch, seq_len_q, d_k)``
    K : shape ``(batch, seq_len_k, d_k)``
    V : shape ``(batch, seq_len_k, d_v)``
    mask : shape broadcastable to ``(..., seq_len_q, seq_len_k)`` or None
           Additive: 0 = allowed, -inf = masked.

    Returns
    -------
    output : shape ``(batch, seq_len_q, d_v)``
    cache  : dict with intermediates for backward
    """
    # TODO: implement the four steps:
    # 1. scores = Q @ K^T / sqrt(d_k)          — use np.swapaxes(K, -1, -2) for K^T
    # 2. if mask is not None: scores += mask
    # 3. weights = softmax(scores, axis=-1)
    # 4. output = weights @ V
    # Return (output, cache) where cache = {"Q", "K", "V", "weights", "mask"}
    raise NotImplementedError


def scaled_dot_product_attention_backward(
    dout: np.ndarray,
    cache: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-head scaled dot-product attention — backward.

    Parameters
    ----------
    dout : gradient w.r.t. output, shape ``(batch, seq_len_q, d_v)``
    cache : dict from forward()

    Returns
    -------
    dQ, dK, dV : each same shape as the corresponding forward input
    """
    # TODO: implement the five steps:
    # 1. dV = weights^T @ dout                  — use np.swapaxes(weights, -1, -2)
    # 2. dweights = dout @ V^T                   — use np.swapaxes(V, -1, -2)
    # 3. dscores = softmax_backward(dweights, weights, axis=-1)
    # 4. dQ = dscores @ K / sqrt(d_k)
    # 5. dK = dscores^T @ Q / sqrt(d_k)          — use np.swapaxes(dscores, -1, -2)
    raise NotImplementedError


# =============================================================================
# Layer 3 — Multi-Head Attention
# =============================================================================

class MultiHeadAttention:
    """Multi-head attention supporting both self- and cross-attention.

    Parameters
    ----------
    d_model : int
        Total model dimension (must be divisible by n_heads).
    n_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout

        # Projection layers
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)

        self._cache: dict | None = None

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """Reshape (batch, seq, d_model) → (batch, n_heads, seq, d_k).

        Steps:
        1. Reshape into (batch, seq, n_heads, d_k)
        2. Transpose to (batch, n_heads, seq, d_k)
        """
        batch, seq_len, _ = x.shape
        x = x.reshape(batch, seq_len, self.n_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """Reshape (batch, n_heads, seq, d_k) → (batch, seq, d_model).

        Steps:
        1. Transpose back to (batch, seq, n_heads, d_k)
        2. Reshape into (batch, seq, d_model)
        """
        batch, _, seq_len, _ = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch, seq_len, self.d_model)

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: np.ndarray | None = None,
        training: bool = True,
    ) -> np.ndarray:
        """Multi-head attention forward.

        - Self-attention:  pass the same tensor for Q, K, V.
        - Cross-attention: pass different tensors.

        Parameters
        ----------
        Q : shape ``(batch, seq_len_q, d_model)``
        K : shape ``(batch, seq_len_k, d_model)``
        V : shape ``(batch, seq_len_k, d_model)``
        mask : optional additive mask, broadcastable to ``(..., seq_len_q, seq_len_k)``
        training : bool — enables dropout when True

        Returns
        -------
        output : shape ``(batch, seq_len_q, d_model)``
        """
        # TODO: implement the six steps:
        # 1. Project Q, K, V using self.W_q.forward(Q), self.W_k.forward(K), self.W_v.forward(V)
        # 2. Split into heads: self._split_heads(Q_proj), etc.
        # 3. Reshape mask if needed — add n_heads dimension:
        #    - if mask.ndim == 2: mask = mask[np.newaxis, np.newaxis, :, :]
        #    - elif mask.ndim == 3: mask = mask[:, np.newaxis, :, :]
        # 4. Apply scaled_dot_product_attention_forward(Q_heads, K_heads, V_heads, mask)
        # 5. Merge heads: self._merge_heads(attn_out)
        # 6. Output projection: self.W_o.forward(output)
        # Store everything in self._cache for backward
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Multi-head attention backward.

        Parameters
        ----------
        dout : gradient w.r.t. output, shape ``(batch, seq_len_q, d_model)``

        Returns
        -------
        dQ, dK, dV : gradients w.r.t. the three inputs
        """
        # TODO: implement the five steps:
        # 1. Gradient through output projection: d_attn_merged = self.W_o.backward(dout)
        # 2. Split back to heads: d_attn_heads = self._split_heads(d_attn_merged)
        # 3. Gradient through attention: scaled_dot_product_attention_backward(d_attn_heads, cache["attn_cache"])
        # 4. Merge head gradients back: self._merge_heads(dQ_heads), etc.
        # 5. Gradient through Q, K, V projections: self.W_q.backward(dQ_proj), etc.
        raise NotImplementedError

    def set_weights_from_torch(self, torch_mha) -> None:
        """Copy weights from a PyTorch ``nn.MultiheadAttention`` for exact comparison.

        The PyTorch module must have been created with ``batch_first=True``.

        This is provided for you — no need to implement.
        """
        import torch

        def to_np(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().numpy().astype(np.float32)

        embed_dim = self.d_model

        # PyTorch stores in_proj_weight as (3*embed_dim, embed_dim)
        in_proj_w = to_np(torch_mha.in_proj_weight)
        in_proj_b = to_np(torch_mha.in_proj_bias)

        # Split into Q, K, V — note the transpose!
        self.W_q.W = in_proj_w[:embed_dim].T.copy()
        self.W_k.W = in_proj_w[embed_dim:2 * embed_dim].T.copy()
        self.W_v.W = in_proj_w[2 * embed_dim:].T.copy()

        self.W_q.b = in_proj_b[:embed_dim].copy()
        self.W_k.b = in_proj_b[embed_dim:2 * embed_dim].copy()
        self.W_v.b = in_proj_b[2 * embed_dim:].copy()

        # Output projection
        self.W_o.W = to_np(torch_mha.out_proj.weight).T.copy()
        self.W_o.b = to_np(torch_mha.out_proj.bias).copy()


# =============================================================================
# Layer 4 — Layer Normalization
# =============================================================================

class LayerNorm:
    """Layer normalisation:  y = (x - μ) / √(σ² + ε) * γ + β.

    Parameters
    ----------
    normalized_shape : int
        Size of the last dimension to normalise over.
    eps : float
        Small constant for numerical stability.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.gamma = np.ones(normalized_shape, dtype=np.float32)
        self.beta = np.zeros(normalized_shape, dtype=np.float32)
        self.eps = eps
        self._cache: dict | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.

        Parameters
        ----------
        x : shape ``(..., normalized_shape)``

        Returns
        -------
        y : same shape as x
        """
        # TODO: implement three steps:
        # 1. mean = np.mean(x, axis=-1, keepdims=True)
        #    var  = np.var(x, axis=-1, keepdims=True)
        # 2. x_norm = (x - mean) / np.sqrt(var + eps)
        # 3. out = gamma * x_norm + beta
        # Store mean, var, x_norm, inv_std(=1/sqrt(var+eps)) in self._cache
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass.

        Parameters
        ----------
        dout : gradient w.r.t. output, same shape as x

        Returns
        -------
        dx : gradient w.r.t. input
        """
        # TODO: implement three steps:
        # 1. dgamma = sum(dout * x_norm, axis=0,...,N-2)  — flatten all except last dim
        #    dbeta  = sum(dout, axis=0,...,N-2)
        # 2. dx_norm = dout * gamma
        # 3. dx = inv_std/N * (N*dx_norm - sum(dx_norm) - x_norm*sum(dx_norm*x_norm))
        #    (flatten to (-1, N) for the sums, then reshape back)
        raise NotImplementedError


# =============================================================================
# Layer 5 — Transformer Attention Block (bonus)
# =============================================================================

class AttentionBlock:
    """A single Transformer block:  LN → MHA (+ residual) → LN → FFN (+ residual).

    This is the canonical pre-LN Transformer block.

    Parameters
    ----------
    d_model : int
    n_heads : int
    d_ff : int
        Hidden dimension of the feed-forward network.
    dropout : float
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        self.ln1 = LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = LayerNorm(d_model)
        # Simple two-layer FFN
        self.ffn_w1 = Linear(d_model, d_ff)
        self.ffn_w2 = Linear(d_ff, d_model)
        self.dropout = dropout
        self._cache: dict | None = None

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None, training: bool = True) -> np.ndarray:
        """Forward pass — pre-LN Transformer block.

        Parameters
        ----------
        x : shape ``(batch, seq_len, d_model)``
        mask : optional attention mask
        training : bool

        Returns
        -------
        out : shape ``(batch, seq_len, d_model)``
        """
        # TODO: implement (bonus challenge)
        # 1. LN → MHA → dropout → residual:  x = x + dropout(mha(ln1(x)))
        # 2. LN → FFN → dropout → residual:  out = x + dropout(ffn_w2(relu(ffn_w1(ln2(x)))))
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass."""
        # TODO: implement (bonus challenge)
        raise NotImplementedError


# =============================================================================
# Quick manual test
# =============================================================================
if __name__ == "__main__":
    print("attention.py loaded successfully.")
    print("Start implementing from Layer 0 and work your way up!")
    print("Run 'python test_attention.py' after each implementation to verify.")