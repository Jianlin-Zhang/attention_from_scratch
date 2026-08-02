"""
Attention Mechanisms from Scratch — Pure NumPy Implementation
=============================================================
Phase 1 of a three-phase project:  NumPy  →  C/C++  →  CUDA.

Layered from low-level utilities up to a full Transformer block, then out into
training, inference and deployment concerns.  Every component is a stub for you
to fill in.  Run the tests alongside to validate step by step:

    conda activate attn-scratch                             # see environment.yml
    python -m unittest test_attention -v                    # everything
    python -m unittest test_attention.TestRoPE -v           # one topic

Conventions
-----------
* Batch-first: all tensors are ``(batch, seq_len, features)``.
* Multi-head tensors are ``(batch, n_heads, seq_len, head_dim)``.
* Masks are **additive**: ``0`` = allowed, ``-inf`` = masked (added before softmax).
* Forward returns a ``cache`` dict of intermediates; backward consumes it. 凡是backward需要, 又无法从输入重算出来的东西, forward就必须存下来
* Parameter grads land on attributes (``self.dW``, ``self.dgamma``, ...);
  input grads are returned.

Reading the stubs
-----------------
Docstrings give you the **contract** (shapes, dtypes, semantics) and, where
useful, a ``Phase 2/3 note`` explaining what this operator turns into once you
drop to C or CUDA.  They deliberately do **not** give you the derivation —
the ``Think about`` prompts are there to be answered by you, not read as hints.
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
    max_in_x = np.max(x, axis=axis, keepdims=True)
    num = np.exp(x - max_in_x)
    den = np.sum(num, axis=axis, keepdims=True)
    return num/den


def softmax_backward(dout: np.ndarray, probs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Gradient of softmax.

    Given  y = softmax(x)  and  dL/dy = dout, returns  dL/dx.

    Formula:  dL/dx_i = y_i * (dout_i - Σ_j dout_j * y_j)
    """
    # TODO: implement the softmax gradient formula above
    # Hint: compute sum(dout * probs) along axis, then probs * (dout - sum_term)
    
    # softmax receive raw logits(n-dim x), and outputs probs(n-dim y)
    # y_deriv_x is n*n Jacobian, y_i*(1 - y_i) when on diag, -y_i*y_j when not on diag
    # bwd return is dout(1*n)*y_deriv_x(n*n)
    return probs * (dout - np.sum(dout*probs, axis=axis, keepdims=True))


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
    
    # row is query for now, col is key(including future teacher ans)
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask = np.where(mask==1, -np.inf, 0.0)
    return np.float32(mask)


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
    
    # what does padding mask do? 
    # different samples may have different valid tokens len, use padding to align them, but use mask to ignore padding pos during attn
    # (batch, n_heads, query_dim, key_dim)
    positions = np.arange(max_len).reshape(1, 1, 1, -1) # (1, 1, 1, max_len)
    valid_expand = valid_lens.reshape(-1, 1, 1, 1) # (batch, 1, 1, 1)
    mask = np.where(positions >= valid_expand, -np.inf, 0.0) # (batch, 1, 1, max_len)
    return np.float32(mask)
    
    
def dropout_forward(x: np.ndarray, p: float, training: bool) -> tuple[np.ndarray, np.ndarray | None]:
    """Inverted dropout: scales kept neurons by 1/(1-p) during training.

    Returns (output, mask).  Mask is ``None`` when not training.
    """
    # TODO: if not training or p <= 0, return (x, None)
    # Otherwise: generate random mask, apply mask, scale by 1/(1-p)
    
    # dropout during training, do nothing during inference
    # [IN]x: input array, do not care its shape, bcz dropout is an element-wise pure manip
    # [IN]p: prob of tossing away randomly
    # [IN]training: a flag to control behavior of this kernel
    # [OUT]output: a same size array as x, after dropout manip
    # [OUT]mask: random mask for manip, return as a note, for backward use. WHEN INFER, return None
    if not training or p <= 0:
        return (x, None)
    random_uniform_arr = np.random.uniform(size=x.shape)
    mask = np.where(random_uniform_arr <= p, 0.0, 1.0).astype(np.float32)
    x_after_drop = (1/(1-p)) * mask * x 
    return (x_after_drop, mask)


def dropout_backward(dout: np.ndarray, mask: np.ndarray | None, p: float) -> np.ndarray:
    """Backward pass for inverted dropout."""
    # TODO: if mask is None, return dout. Otherwise scale by mask / (1-p)
    if mask is None:
        return dout
    else:
        return (1/(1-p)) * mask * dout

# -----------------------------------------------------------------------------
# Layer 0b — Explicit-loop kernels  (the Phase 2 / Phase 3 bridge)
# -----------------------------------------------------------------------------
# ``A @ B`` is one character in NumPy and ~200 lines in CUDA.  Writing matmul
# twice here — once naively, once tiled — is what makes Phase 2 a *port* rather
# than a from-scratch rewrite.  Both must agree with ``np.matmul`` bit-for-bit
# closely enough to pass the tests.

def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """2-D matrix multiply written with **explicit Python loops**.

    Parameters
    ----------
    A : shape ``(M, K)``
    B : shape ``(K, N)``

    Returns
    -------
    C : shape ``(M, N)``, dtype float32

    Rules
    -----
    You may use ``np.zeros`` to allocate the output and scalar indexing to read
    and write elements.  You may **not** use ``@``, ``np.matmul``, ``np.dot``,
    ``np.einsum``, or vectorised slicing that does the reduction for you.
    Three nested loops, scalar accumulate.

    Phase 2/3 note
    --------------
    This is the exact loop nest you will write in C.  The loop order you pick
    (i-j-k vs i-k-j vs k-i-j) changes nothing here but changes C performance by
    several times.

    Think about
    -----------
    * Which of the three loops carries the reduction (i.e. accumulates)?
    * For each loop order, which array is walked contiguously and which is
      strided?  Write the answer down now — you will verify it in Phase 2.
    """
    # TODO: three nested loops
    
    # for now, we just use the simplest i-j-k form
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N), np.float32)
    for i in range(M): # output mat row-i
        for j in range(N): # output mat col-j
            elem = 0
            for k in range(K): # A row-i(size 1*k), dot with B col-j(size k*1)
                elem += A[i, k]*B[k, j]
            C[i, j] = elem
    return C


def matmul_tiled(A: np.ndarray, B: np.ndarray, tile: int = 32) -> np.ndarray:
    """2-D matrix multiply with **cache blocking / tiling**.

    Parameters
    ----------
    A : shape ``(M, K)``
    B : shape ``(K, N)``
    tile : side length of the square tile

    Returns
    -------
    C : shape ``(M, N)``, dtype float32

    Rules
    -----
    Loop over tiles of M, N and K; inside a tile you may use ``@`` on the small
    sub-blocks (that stands in for "the block is now in shared memory / registers").
    Must be correct when ``M``, ``N``, ``K`` are **not** multiples of ``tile``.

    Phase 2/3 note
    --------------
    In CUDA the M/N tile loops become the grid, the K tile loop stays a loop
    inside the kernel, and the sub-block ``@`` becomes a shared-memory
    accumulation. Handling the ragged edge tile is exactly the CUDA boundary
    check you will need.

    Think about
    -----------
    * Why must the K loop be the *innermost* of the three tile loops if you want
      to write each output tile only once?
    * How many times is each element of A read, as a function of ``tile``?
    """
    # TODO: tile over M, N, K; accumulate into the output block
    raise NotImplementedError


def online_softmax_update(
    m_prev: np.ndarray,
    l_prev: np.ndarray,
    acc_prev: np.ndarray,
    scores_block: np.ndarray,
    v_block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One streaming step of the *online softmax* recurrence.

    Processes a new block of scores without ever materialising the full score
    row, maintaining a running (max, sum-of-exp, weighted-value) state.

    Parameters
    ----------
    m_prev : shape ``(..., seq_q, 1)`` — running row max so far. Use ``-inf`` to start.
    l_prev : shape ``(..., seq_q, 1)`` — running sum of exponentials so far. Use ``0`` to start.
    acc_prev : shape ``(..., seq_q, d_v)`` — running unnormalised output. Use ``0`` to start.
    scores_block : shape ``(..., seq_q, block_k)`` — new scores, already scaled and masked.
    v_block : shape ``(..., block_k, d_v)`` — the V rows matching ``scores_block``.

    Returns
    -------
    m_new, l_new, acc_new : same shapes as the ``*_prev`` inputs.

    Contract
    --------
    After streaming every block, ``acc / l`` must equal the ordinary
    ``softmax(scores) @ V``.  The test drives this function block by block and
    compares against your Layer 2 implementation.

    Phase 2/3 note
    --------------
    This recurrence is the whole reason FlashAttention exists: it makes attention
    O(seq) in memory instead of O(seq²), which is what lets the score block live
    in CUDA shared memory. Layer 9 builds on this.

    Think about
    -----------
    * When the new block's max exceeds ``m_prev``, the previously accumulated
      terms were exponentiated against the *old* max. What single multiplicative
      correction repairs both ``l_prev`` and ``acc_prev``?
    * Why is it safe (no overflow) even if a whole block is ``-inf``?
    * Does this recurrence need to be numerically identical to the one-shot
      softmax, or only close? Check what the test tolerance actually demands.
    """
    # TODO: rescale the running state to the new max, then fold in the new block
    raise NotImplementedError


# -----------------------------------------------------------------------------
# Layer 0c — Activations
# -----------------------------------------------------------------------------

def gelu_forward(x: np.ndarray, approximate: str = "tanh") -> tuple[np.ndarray, dict]:
    """GELU activation.

    Parameters
    ----------
    x : any shape
    approximate : ``"tanh"`` for the tanh approximation (what GPT-2 shipped),
                  ``"none"`` for the exact erf form.

    Returns
    -------
    out : same shape as x
    cache : dict with whatever backward needs (must include ``x`` and ``approximate``)

    Contract
    --------
    Exact form uses the Gaussian CDF; the tanh form is the classic
    ``0.5x(1+tanh(c(x+0.044715x³)))`` with ``c = sqrt(2/pi)``. The two differ by
    ~1e-3 at worst — the test checks each against the matching PyTorch mode, so
    you cannot pass by implementing only one.

    Phase 2/3 note
    --------------
    ``erf`` is a libm call; ``tanh`` is a few multiplies. On GPU the approximation
    exists because transcendental throughput is the bottleneck in an elementwise
    kernel. In Phase 3 you will measure whether that is still true.

    Think about
    -----------
    * GELU is not monotonic near ``x ≈ -0.75``. What does that imply about the
      sign of the gradient there, and does your backward reproduce it?
    """
    # TODO: implement both branches
    raise NotImplementedError


def gelu_backward(dout: np.ndarray, cache: dict) -> np.ndarray:
    """Gradient of GELU. Must handle both ``approximate`` modes.

    Think about
    -----------
    * For the exact form the derivative contains both the Gaussian CDF and PDF.
      Which one comes from the product rule and which from differentiating the CDF?
    """
    # TODO
    raise NotImplementedError


def silu_forward(x: np.ndarray) -> tuple[np.ndarray, dict]:
    """SiLU / Swish:  ``x * sigmoid(x)``.

    Returns ``(out, cache)``.  Watch overflow in ``exp`` for very negative ``x``.
    """
    # TODO
    raise NotImplementedError


def silu_backward(dout: np.ndarray, cache: dict) -> np.ndarray:
    """Gradient of SiLU.

    Think about
    -----------
    * You can express this purely in terms of ``sigmoid(x)`` and ``x``. Doing so
      means the forward only has to cache one array instead of two — which
      matters in Phase 3, where cached activations are the memory budget.
    """
    # TODO
    raise NotImplementedError


def swiglu_forward(x_gate: np.ndarray, x_up: np.ndarray) -> tuple[np.ndarray, dict]:
    """SwiGLU:  ``silu(x_gate) * x_up``  (LLaMA / PaLM feed-forward activation).

    Parameters
    ----------
    x_gate : shape ``(..., d_ff)`` — output of the gate projection
    x_up   : shape ``(..., d_ff)`` — output of the up projection

    Returns
    -------
    out : shape ``(..., d_ff)``
    cache : dict for backward

    Think about
    -----------
    * A SwiGLU FFN needs three weight matrices where a ReLU FFN needs two.
      To keep the parameter count equal, what does ``d_ff`` have to become?
      (This is why LLaMA uses 8/3·d_model rounded, not 4·d_model.)
    """
    # TODO
    raise NotImplementedError


def swiglu_backward(dout: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of SwiGLU. Returns ``(dx_gate, dx_up)``."""
    # TODO
    raise NotImplementedError


# -----------------------------------------------------------------------------
# Layer 0d — More masks
# -----------------------------------------------------------------------------

def create_sliding_window_mask(seq_len: int, window_size: int) -> np.ndarray:
    """Causal mask restricted to a sliding local window (Mistral / Longformer style).

    Parameters
    ----------
    seq_len : sequence length
    window_size : how many positions a query may attend to **including itself**.
                  ``window_size=1`` → each query sees only itself.
                  ``window_size >= seq_len`` → degenerates to a plain causal mask.

    Returns
    -------
    mask : shape ``(seq_len, seq_len)``, float32.
           Query ``i`` may attend to key ``j`` if ``i - window_size < j <= i``.
           Allowed → 0, disallowed → ``-inf``.

    Phase 2/3 note
    --------------
    A local window means most score tiles are entirely masked. In Phase 3 the
    win is not the mask itself but *skipping those tiles' loads altogether* —
    the same trick block-sparse attention kernels use.

    Think about
    -----------
    * Every row has at most ``window_size`` visible entries, so no row can ever
      be fully masked. Why does that matter for softmax?
    """
    # TODO
    # every query can only "see" its prev WINDOW_SIZE keys(attn to), make O(seq^2)->O(seq*Const)
    # HOW DOES IT SOLVE "long-horizon"? every new layer's key is prev layer's mixed output, it covers a "sliding ans"
    query_row = np.arange(seq_len).reshape(-1, 1) # (seq_len, 1)
    query_col = np.arange(seq_len).reshape(1, -1) # (1, seq_len)
    diff_arr = query_row - query_col # (seq_len, seq_len), every elem is i-j(query row - key col)
    windowed_mask = np.where((diff_arr >= 0) & (diff_arr < window_size), 0.0, -np.inf).astype(np.float32)
    return windowed_mask
    

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
        y = x @ self.W + self.b
        self._cache = dict({'x': x})
        return y

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
        
        # simple reference: https://cs231n.stanford.edu/handouts/linear-backprop.pdf
        # dW, very complex, need to draw and calc the deriv-link
        x_flat = self._cache['x'].reshape(-1, self._cache['x'].shape[-1]) # (..., in_features), x might be (batch, seq_len, in_dim)
        dout_flat = dout.reshape(-1, dout.shape[-1]) # (..., out_features)
        self.dW = x_flat.transpose() @ dout_flat # (in_features, out_features)
        # db, multiplier is an EYE
        self.db = np.sum(dout_flat, axis=0) # sum along ... dim, get (out_features,)
        # dx
        dx = dout @ self.W.transpose() # (..., in_features)
        return dx

class Embedding:
    """Token embedding lookup table.

    Parameters
    ----------
    num_embeddings : vocabulary size
    embedding_dim : d_model
    padding_idx : optional row that must always stay all-zero **and** must
                  receive zero gradient.

    Attributes
    ----------
    W : shape ``(num_embeddings, embedding_dim)``
    dW : same shape, filled by ``backward``
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.W: np.ndarray = np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * 0.02
        if padding_idx is not None:
            self.W[padding_idx] = 0.0
        self.dW: np.ndarray | None = None
        self._cache: dict | None = None

    def forward(self, idx: np.ndarray) -> np.ndarray:
        """Look up rows of ``W``.

        Parameters
        ----------
        idx : integer array of shape ``(batch, seq_len)``

        Returns
        -------
        out : shape ``(batch, seq_len, embedding_dim)``

        Think about
        -----------
        * This is a gather. Is the returned array a view or a copy, and does the
          answer change your backward?
        """
        # TODO
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> None:
        """Accumulate ``self.dW``. There is no input gradient (indices are discrete).

        Parameters
        ----------
        dout : shape ``(batch, seq_len, embedding_dim)``

        Contract
        --------
        ``self.dW`` must be a fresh array of ``W``'s shape each call, and
        ``dW[padding_idx]`` must be exactly zero.

        Phase 2/3 note
        --------------
        Repeated tokens in a batch all write to the *same* row, so this is a
        scatter-**add**, not a scatter. ``dW[idx] += dout`` silently gives the
        wrong answer in NumPy for duplicate indices — find out why, because the
        identical hazard in CUDA is what forces ``atomicAdd``.

        Think about
        -----------
        * Which NumPy function performs a scatter-add correctly?
        * A vocab of 50k × d_model 768 makes dW 150 MB of mostly zeros. What is
          the sparse alternative, and why do frameworks offer it as an option?
        """
        # TODO
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
    # 1. calc attn scores
    # 2. apply mask on scores
    # 3. normalize scores
    # 4. output
    # Return (output, cache) where cache = {"Q", "K", "V", "mask"}
    
    attn_score = (Q @ np.swapaxes(K, -1, -2)) / np.sqrt(K.shape[-1]) # (batch, seq_len_q, seq_len_k)
    if mask is not None:
        attn_score += mask
    output = softmax(attn_score, -1) @ V # (batch, seq_len_q, d_v)
    cache = dict({
        'Q': Q,
        'K': K,
        'V': V,
        'mask': mask,
    })
    return (output, cache)
    
    
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

def repeat_kv(x: np.ndarray, n_rep: int) -> np.ndarray:
    """Expand KV heads to match query heads — the mechanism behind MQA / GQA.

    Parameters
    ----------
    x : shape ``(batch, n_kv_heads, seq_len, head_dim)``
    n_rep : ``n_q_heads // n_kv_heads``.  ``n_rep == 1`` → plain MHA;
            ``n_kv_heads == 1`` → MQA; in between → GQA.

    Returns
    -------
    out : shape ``(batch, n_kv_heads * n_rep, seq_len, head_dim)``

    Contract
    --------
    Grouping is **contiguous, not interleaved**: KV head 0 must land at output
    heads ``0 .. n_rep-1``, KV head 1 at ``n_rep .. 2*n_rep-1``, and so on.
    (Getting this backwards produces plausible-looking garbage that only shows
    up as a quality regression — the test pins the layout down.)

    Phase 2/3 note
    --------------
    The whole point of GQA is *not* saving parameters — it is shrinking the KV
    cache, which is what bounds inference batch size. In CUDA you never actually
    materialise this expansion: the kernel just has several query heads index the
    same KV tile. Implementing it as a real copy here, then removing the copy in
    Phase 3, is a useful before/after.

    Think about
    -----------
    * How much smaller is the KV cache for LLaMA-2-70B (64 q heads, 8 kv heads)?
    * ``np.repeat`` and ``np.tile`` differ exactly in the interleaving question
      above. Which one do you want?
    """
    # TODO
    raise NotImplementedError


def repeat_kv_backward(dout: np.ndarray, n_rep: int) -> np.ndarray:
    """Gradient of :func:`repeat_kv`.

    Parameters
    ----------
    dout : shape ``(batch, n_kv_heads * n_rep, seq_len, head_dim)``
    n_rep : same value used in the forward

    Returns
    -------
    dx : shape ``(batch, n_kv_heads, seq_len, head_dim)``

    Think about
    -----------
    * A value used in ``n_rep`` places contributes to the loss through all of
      them. Broadcast forward ⇒ what operation backward?
    """
    # TODO
    raise NotImplementedError


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

    def forward_incremental(
        self,
        x_step: np.ndarray,
        kv_cache: "KVCache",
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Single-token decode step using a KV cache — the inference-time path.

        Parameters
        ----------
        x_step : shape ``(batch, 1, d_model)`` — the newly generated token only.
        kv_cache : a :class:`KVCache` holding all previous keys and values.
                   This call must append the current step's K/V to it.
        mask : usually ``None``. A single query attending to all cached keys needs
               no causal mask — convince yourself why before you pass one.

        Returns
        -------
        out : shape ``(batch, 1, d_model)``

        Contract
        --------
        Running prefill-then-decode token by token must produce **the same
        outputs** as one causal ``forward`` over the whole sequence. The test
        asserts this equivalence; it is the single most valuable test in the file
        because almost every hand-rolled KV cache is subtly wrong at first.

        Phase 2/3 note
        --------------
        Prefill is compute-bound (a big GEMM); decode is memory-bound (a
        skinny GEMV that re-reads the entire cache per token). Nearly every
        inference optimisation — paged cache, speculative decoding, flash-decoding
        — targets that asymmetry. You cannot feel it in NumPy, but you can
        already *count* the bytes moved per token.

        Think about
        -----------
        * ``Q`` has ``seq_len_q == 1`` while ``K``/``V`` have ``seq_len_k == cache_len``.
          Which of your existing functions already handles that without changes?
        * No backward pass is needed here. What does that let you skip storing?
        * If positions matter (RoPE), where does the position index come from now
          that ``x_step`` no longer knows where it sits in the sequence?
        """
        # TODO
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


class RMSNorm:
    """Root-mean-square normalisation (LLaMA / T5 style):  no mean subtraction, no bias.

    Parameters
    ----------
    normalized_shape : size of the last dimension
    eps : numerical floor

    Attributes
    ----------
    gamma : shape ``(normalized_shape,)``, init to ones
    dgamma : filled by ``backward``
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        self.gamma = np.ones(normalized_shape, dtype=np.float32)
        self.eps = eps
        self.dgamma: np.ndarray | None = None
        self._cache: dict | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Scale each row by the reciprocal of its RMS, then apply ``gamma``.

        Parameters
        ----------
        x : shape ``(..., normalized_shape)``

        Returns
        -------
        y : same shape as x

        Contract
        --------
        ``eps`` goes **inside** the square root, added to the mean of squares —
        matching PyTorch. Placing it outside changes results at small magnitudes
        and the test will catch it.

        Note: ``torch.nn.functional.rms_norm`` does not exist in torch 2.2, so the
        test builds its own reference and differentiates it with autograd.

        Think about
        -----------
        * LayerNorm is invariant to adding a constant to every element of a row;
          RMSNorm is not. Which invariance did we give up, and why did it turn out
          not to matter empirically?
        """
        # TODO
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass. Sets ``self.dgamma``, returns ``dx``.

        Think about
        -----------
        * LayerNorm backward has two correction terms; RMSNorm backward has one.
          Which of the two disappeared, and what in the forward pass was
          responsible for it?
        * The row scale depends on every element of the row, so ``dx`` is not
          elementwise. What is the rank of the correction you have to subtract?
        """
        # TODO
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
# Layer 6 — Positional information
# =============================================================================
# Attention is permutation-equivariant: shuffle the tokens and the outputs shuffle
# with them. Everything in this layer exists to break that symmetry, and the three
# approaches break it in interestingly different places (input, input, scores).

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Fixed sinusoidal positional encoding from the original Transformer paper.

    Parameters
    ----------
    seq_len : number of positions
    d_model : must be even

    Returns
    -------
    pe : shape ``(seq_len, d_model)``, float32.
         Even channels are sines, odd channels are cosines, and channel pair
         ``i`` uses wavelength ``10000^(2i/d_model)``.

    Think about
    -----------
    * The paper's stated motivation is that ``PE[pos+k]`` is a *linear* function of
      ``PE[pos]`` for fixed ``k``. Verify that numerically once you have it — it is
      the seed of the idea RoPE takes to its conclusion.
    * Why compute the frequencies in log-space rather than as a direct power?
    """
    # TODO
    raise NotImplementedError


def rope_precompute_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0) -> tuple[np.ndarray, np.ndarray]:
    """Precompute the rotation table for Rotary Position Embedding.

    Parameters
    ----------
    head_dim : per-head dimension, must be even
    max_seq_len : longest position you will ever need
    base : the ``theta`` hyperparameter (10000 originally; long-context models
           raise it — that is all "RoPE scaling" is, at bottom)

    Returns
    -------
    cos, sin : each shape ``(max_seq_len, head_dim // 2)``, float32

    Contract
    --------
    Frequency ``i`` (``0 <= i < head_dim/2``) is ``base ** (-2i / head_dim)``,
    and entry ``[p, i]`` is the cosine/sine of that frequency times position ``p``.

    Think about
    -----------
    * Why is this table position-only and not data-dependent — i.e. why can it be
      computed once at model load and never again?
    """
    # TODO
    raise NotImplementedError


def rope_apply(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, offset: int = 0) -> np.ndarray:
    """Apply rotary embedding to a Q or K tensor.

    Parameters
    ----------
    x : shape ``(batch, n_heads, seq_len, head_dim)``
    cos, sin : tables from :func:`rope_precompute_freqs`
    offset : position of ``x[:, :, 0]`` in the full sequence. ``0`` during
             training/prefill; ``cache_len`` during incremental decoding.

    Returns
    -------
    out : same shape as ``x``

    Contract — pairing convention
    -----------------------------
    Use the **split-half** convention (GPT-NeoX / HuggingFace LLaMA): channel
    ``i`` is paired with channel ``i + head_dim/2``, and the pair is rotated by
    the angle whose cosine/sine are ``cos[pos, i], sin[pos, i]``. The alternative
    *interleaved* convention ``(0,1), (2,3), ...`` is what the original RoPE paper
    describes; both are used in the wild and they are **not** interchangeable.
    The test pins split-half.

    Phase 2/3 note
    --------------
    RoPE is applied to Q and K only, never V, and it is a cheap elementwise kernel
    over a strided access pattern. In Phase 3 it is a prime candidate for fusing
    into the QKV projection epilogue rather than launching its own kernel.

    Think about
    -----------
    * A 2-D rotation is length-preserving. So what is conserved about every
      ``q · k`` after you rotate both?
    * Work out why the dot product ends up depending on ``m - n`` rather than on
      ``m`` and ``n`` separately. That single fact is the whole justification for
      calling this *relative* position encoding.
    * Why does the ``offset`` argument make or break KV-cached generation?
    """
    # TODO
    raise NotImplementedError


def rope_backward(dout: np.ndarray, cos: np.ndarray, sin: np.ndarray, offset: int = 0) -> np.ndarray:
    """Gradient of :func:`rope_apply`.

    Think about
    -----------
    * The forward is a fixed orthogonal linear map applied per position. What is
      the transpose of a 2-D rotation by angle ``t``?
    * You should be able to write this by changing one sign relative to the
      forward. If you find yourself building a Jacobian, step back.
    """
    # TODO
    raise NotImplementedError


def alibi_slopes(n_heads: int) -> np.ndarray:
    """Per-head slopes for ALiBi (Attention with Linear Biases).

    Parameters
    ----------
    n_heads : number of attention heads

    Returns
    -------
    slopes : shape ``(n_heads,)``, float32, strictly decreasing, all positive.

    Contract
    --------
    For power-of-two ``n_heads`` the slopes are the geometric sequence with ratio
    ``r = 2 ** (-8 / n_heads)``, i.e. ``r, r^2, ..., r^n_heads``. (So ``n_heads=8``
    gives exactly ``1/2, 1/4, ..., 1/256``.) For non-powers of two the reference
    implementation falls back to interleaving two such sequences — handle it
    however you like, the test only pins the power-of-two case.

    Think about
    -----------
    * Heads with a large slope are forced to be extremely local. What division of
      labour across heads does this create?
    """
    # TODO
    raise NotImplementedError


def alibi_bias(n_heads: int, seq_q: int, seq_k: int, offset: int = 0) -> np.ndarray:
    """Build the additive ALiBi bias tensor.

    Parameters
    ----------
    n_heads, seq_q, seq_k : dimensions
    offset : position of the first query, for the decoding case

    Returns
    -------
    bias : shape ``(1, n_heads, seq_q, seq_k)``, float32.
           Head ``h``'s entry for query ``i`` and key ``j`` penalises distance:
           it is ``-slopes[h]`` times the gap between the two positions.

    Contract
    --------
    This is added to the scores *alongside* the causal mask, not instead of it.
    Values must be ``<= 0``, and exactly ``0`` where query and key positions match.

    Think about
    -----------
    * ALiBi needs no learned parameters and no embedding table, and it
      extrapolates to sequences longer than those seen in training. Given that,
      why did the field standardise on RoPE instead?
    * Where in the pipeline does ALiBi intervene, compared to RoPE? (One touches
      Q/K, the other touches the score matrix.) Which is cheaper to fuse into a
      FlashAttention kernel?
    """
    # TODO
    raise NotImplementedError


# =============================================================================
# Layer 7 — Training machinery
# =============================================================================

def cross_entropy_forward(
    logits: np.ndarray,
    targets: np.ndarray,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> tuple[float, dict]:
    """Softmax cross-entropy loss, fused for numerical stability.

    Parameters
    ----------
    logits : shape ``(N, vocab)`` or ``(batch, seq_len, vocab)``
    targets : integer array of shape ``logits.shape[:-1]``. Entries equal to
              ``ignore_index`` contribute nothing to the loss or the gradient.
    ignore_index : label value to skip (padding positions)
    label_smoothing : in ``[0, 1)``. ``0`` disables it.

    Returns
    -------
    loss : Python float — mean over the **non-ignored** positions only.
    cache : dict for backward

    Contract
    --------
    Matches ``F.cross_entropy(..., reduction="mean", ignore_index=..., label_smoothing=...)``.
    Note the denominator: it is the count of valid targets, not ``N``.

    Phase 2/3 note
    --------------
    "Fused" is the operative word. A 50k-vocab logit tensor for a 4k-token batch is
    800 MB in fp32; materialising the softmax probabilities *and* their gradient
    separately doubles that. Real implementations fuse forward and backward and
    never store the probabilities — which is why this signature hands you a cache
    and lets you decide what goes in it.

    Think about
    -----------
    * Why must the log-sum-exp be computed with the max subtracted, given that
      logits routinely reach ±30?
    * With label smoothing ``e``, the target distribution puts most mass on the
      true class and spreads ``e`` over the rest. Over *all* classes or all
      *other* classes? PyTorch made one specific choice; find it, because it
      shifts the loss by a constant and the test compares absolute values.
    """
    # TODO
    raise NotImplementedError


def cross_entropy_backward(cache: dict) -> np.ndarray:
    """Gradient w.r.t. ``logits``. Same shape as the forward's ``logits``.

    Think about
    -----------
    * The composition of softmax and cross-entropy has a famously simple gradient.
      Derive it once by hand so you can recognise it on sight; then work out what
      the ``1/n_valid`` factor and ``ignore_index`` handling do to it.
    * Ignored positions must get exactly zero gradient, not merely small.
    """
    # TODO
    raise NotImplementedError


def clip_grad_norm(grads: list[np.ndarray], max_norm: float, norm_type: float = 2.0) -> float:
    """Clip a list of gradients **in place** by their global norm.

    Parameters
    ----------
    grads : list of arrays, modified in place
    max_norm : threshold
    norm_type : ``2.0`` for L2, ``np.inf`` for max-abs

    Returns
    -------
    total_norm : the global norm computed **before** clipping (this is the number
                 worth logging during training).

    Contract
    --------
    Matches ``torch.nn.utils.clip_grad_norm_``: one scale factor derived from the
    norm over *all* tensors treated as one concatenated vector, applied uniformly.
    Do not clip per-tensor. If ``total_norm <= max_norm``, leave the arrays alone.

    Think about
    -----------
    * Per-tensor clipping would change the *direction* of the update; global
      clipping only changes its length. Why does that distinction matter?
    * What does a sudden spike in the returned value usually tell you about the
      current batch?
    """
    # TODO
    raise NotImplementedError


def lr_cosine_with_warmup(
    step: int,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> float:
    """Linear warmup followed by cosine decay — the default LLM schedule.

    Parameters
    ----------
    step : 0-based current step
    base_lr : peak learning rate, reached at the end of warmup
    warmup_steps : linear ramp length. ``0`` means no warmup.
    total_steps : total training steps
    min_lr : floor that the cosine decays to

    Returns
    -------
    lr : learning rate for this step

    Contract
    --------
    ``step = 0`` during warmup must give a **non-zero** lr (off-by-one here is the
    classic bug: a zero first step is harmless, shifting the whole schedule is
    not). At ``step >= total_steps``, return ``min_lr``.

    Think about
    -----------
    * Why does a Transformer trained with Adam need warmup at all? What is Adam's
      second-moment estimate doing during the first few dozen steps?
    """
    # TODO
    raise NotImplementedError


class SGD:
    """SGD with optional momentum, Nesterov, and weight decay.

    Parameters
    ----------
    params : list of arrays to be updated **in place**
    lr, momentum, weight_decay, nesterov : standard meanings

    Contract
    --------
    Matches ``torch.optim.SGD``, including its specific momentum-buffer
    initialisation on the very first step (PyTorch's choice differs from some
    textbook formulations, and the test compares against PyTorch step by step).
    """

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 1e-2,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.buffers: list[np.ndarray | None] = [None] * len(params)

    def step(self, grads: list[np.ndarray]) -> None:
        """Apply one update. ``grads[i]`` corresponds to ``self.params[i]``.

        Think about
        -----------
        * In plain SGD, weight decay and L2 regularisation are the same thing.
          Note that claim — you will need it to appreciate why AdamW exists.
        """
        # TODO
        raise NotImplementedError


class AdamW:
    """AdamW: Adam with **decoupled** weight decay.

    Parameters
    ----------
    params : list of arrays updated in place
    lr : step size
    betas : ``(beta1, beta2)`` exponential decay rates for the two moments
    eps : denominator floor
    weight_decay : decoupled decay coefficient

    Contract
    --------
    Matches ``torch.optim.AdamW``, including bias correction on both moments and a
    1-based step counter. Both moment buffers start at zero.

    Phase 2/3 note
    --------------
    Optimiser state is two extra fp32 copies of every parameter — for a 7B model
    that is 56 GB, which is why ZeRO sharding and 8-bit optimisers exist. Also
    note this is a pure elementwise kernel over billions of elements: entirely
    memory-bound, and a natural fusion target in Phase 3.

    Think about
    -----------
    * Where does the decay term enter, relative to the division by
      ``sqrt(v) + eps``? Draw both possible orderings; one is Adam+L2 and the
      other is AdamW, and that difference is the entire paper.
    * Why do practitioners exclude biases and LayerNorm gains from weight decay?
    * What do the bias-correction factors approach as ``t`` grows, and what does
      that tell you about when they matter?
    """

    def __init__(
        self,
        params: list[np.ndarray],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m: list[np.ndarray] = [np.zeros_like(p) for p in params]
        self.v: list[np.ndarray] = [np.zeros_like(p) for p in params]

    def step(self, grads: list[np.ndarray]) -> None:
        """Apply one update. Remember to advance ``self.t`` exactly once per call."""
        # TODO
        raise NotImplementedError


# =============================================================================
# Layer 8 — Inference
# =============================================================================

class KVCache:
    """Pre-allocated key/value cache for autoregressive decoding.

    Parameters
    ----------
    batch, n_kv_heads, max_seq_len, head_dim : buffer dimensions

    Attributes
    ----------
    K, V : shape ``(batch, n_kv_heads, max_seq_len, head_dim)``, pre-allocated
    length : how many positions are currently valid

    Design note
    -----------
    Pre-allocating and tracking a length — rather than concatenating each step — is
    the whole point. Concatenation reallocates and copies the entire cache every
    token, turning generation into O(seq^2) memory traffic.

    Phase 2/3 note
    --------------
    This flat pre-allocated buffer wastes memory whenever sequences in a batch have
    different lengths, and it caps you at ``max_seq_len``. Fixing that is what
    PagedAttention (vLLM) does, with a block table mapping logical positions to
    physical pages. Feel the problem here first.
    """

    def __init__(self, batch: int, n_kv_heads: int, max_seq_len: int, head_dim: int, dtype=np.float32):
        self.batch = batch
        self.n_kv_heads = n_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.K = np.zeros((batch, n_kv_heads, max_seq_len, head_dim), dtype=dtype)
        self.V = np.zeros((batch, n_kv_heads, max_seq_len, head_dim), dtype=dtype)
        self.length = 0

    def append(self, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Write new keys/values at the current position and return the valid prefix.

        Parameters
        ----------
        k, v : shape ``(batch, n_kv_heads, n_new, head_dim)``.
               ``n_new > 1`` during prefill, ``n_new == 1`` during decode.

        Returns
        -------
        K_valid, V_valid : shape ``(batch, n_kv_heads, self.length, head_dim)``
                           **after** the append.

        Contract
        --------
        Raise if the append would exceed ``max_seq_len``. The returned arrays must
        cover only the valid region — never the whole padded buffer, or the zeros
        will silently take part in the softmax.

        Think about
        -----------
        * Should you return a view or a copy? A view is free but aliases a buffer
          you will mutate on the next step. Which does the caller actually need?
        """
        # TODO
        raise NotImplementedError

    def reset(self) -> None:
        """Reset for a new sequence.

        Think about
        -----------
        * Do you actually have to zero the buffers, or only the length? What
          invariant makes the cheap answer safe?
        """
        # TODO
        raise NotImplementedError


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by ``1 / temperature``.

    Parameters
    ----------
    logits : shape ``(..., vocab)``
    temperature : ``> 0``. ``temperature == 0`` is conventionally treated as
                  greedy — decide whether you handle that here or in the caller,
                  and document your choice.

    Think about
    -----------
    * What are the limits of the resulting distribution as ``temperature -> 0``
      and as ``temperature -> inf``?
    * Why is this applied to logits rather than to the probabilities?
    """
    # TODO
    raise NotImplementedError


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Keep only the ``k`` largest logits per row; set the rest to ``-inf``.

    Parameters
    ----------
    logits : shape ``(..., vocab)``
    k : if ``k <= 0`` or ``k >= vocab``, return the logits unchanged.

    Returns
    -------
    filtered : same shape, a new array (do not mutate the input)

    Think about
    -----------
    * A full sort is O(V log V) per generated token for V ~ 128k. What does
      ``np.argpartition`` give you instead, and why is it sufficient here?
    """
    # TODO
    raise NotImplementedError


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """Nucleus sampling filter: keep the smallest set of tokens whose cumulative
    probability reaches ``p``, and mask the rest to ``-inf``.

    Parameters
    ----------
    logits : shape ``(..., vocab)``
    p : in ``(0, 1]``. ``p >= 1`` returns the logits unchanged.

    Returns
    -------
    filtered : same shape, new array

    Contract
    --------
    The token that *crosses* the threshold is **kept**, so the retained mass is
    always ``>= p`` and at least one token always survives. That last part is not
    optional: a single token can exceed ``p`` on its own.

    Think about
    -----------
    * The off-by-one at the crossing token is the most common bug here, and it is
      invisible except at small ``p``. Write down which comparison you use before
      you code it.
    * Why is top-p usually preferred over top-k for open-ended generation? Think
      about how the shape of the distribution differs between an unambiguous next
      token and a genuinely open one.
    """
    # TODO
    raise NotImplementedError


def sample_from_logits(logits: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Draw one token per row from ``softmax(logits)``.

    Parameters
    ----------
    logits : shape ``(batch, vocab)``, possibly already filtered to contain ``-inf``
    rng : pass a seeded ``np.random.default_rng`` for reproducibility

    Returns
    -------
    idx : shape ``(batch,)``, dtype int64

    Think about
    -----------
    * Implement it with one uniform draw plus a cumulative sum (inverse-CDF
      sampling). That is what a GPU kernel does, because it needs no per-category
      loop with rejection.
    * ``-inf`` logits must be drawn with probability exactly zero. Does your
      cumsum approach guarantee that, including at floating-point boundaries?
    """
    # TODO
    raise NotImplementedError


def greedy_select(logits: np.ndarray) -> np.ndarray:
    """Argmax over the last axis. Returns shape ``logits.shape[:-1]``, dtype int64.

    Trivial on purpose — it is the baseline that the samplers above must reduce to
    at ``temperature -> 0`` and at ``k = 1``.
    """
    # TODO
    raise NotImplementedError


# =============================================================================
# Layer 9 — FlashAttention (tiled, memory-efficient)
# =============================================================================
# This is the centre of gravity of the whole project. The same algorithm carries
# through all three phases:
#   Phase 1 (here): tile loops in NumPy, proving numerical equivalence with Layer 2.
#   Phase 2 (C):    the tile loops become real loops over real buffers.
#   Phase 3 (CUDA): the Q tile becomes a thread block, the K/V tile lives in
#                   shared memory, and the online-softmax state lives in registers.
# Do Layer 2 first — this must be validated against it.

def flash_attention_forward(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    block_q: int = 32,
    block_k: int = 32,
    causal: bool = False,
) -> tuple[np.ndarray, dict]:
    """Tiled attention forward that never materialises the full score matrix.

    Parameters
    ----------
    Q : shape ``(batch, n_heads, seq_q, d_k)``
    K : shape ``(batch, n_heads, seq_k, d_k)``
    V : shape ``(batch, n_heads, seq_k, d_v)``
    block_q, block_k : tile sizes. Must work for any values, including ones that
                       do not divide the sequence lengths.
    causal : apply causal masking *inside* the tile loop

    Returns
    -------
    output : shape ``(batch, n_heads, seq_q, d_v)``
    cache : dict for backward. Must contain ``Q, K, V, output`` and the per-row
            log-sum-exp ``L`` of shape ``(batch, n_heads, seq_q, 1)``.

    Hard requirement
    ----------------
    At no point may an array of shape ``(..., seq_q, seq_k)`` exist. Peak extra
    memory must be ``O(block_q * block_k)``, not ``O(seq_q * seq_k)``. Writing the
    naive version and then "tiling" it by slicing a full score matrix defeats the
    entire exercise.

    Contract
    --------
    ``output`` must match :func:`scaled_dot_product_attention_forward` to ~1e-5.
    Storing ``L`` instead of the weights is what makes the backward possible —
    that trade of recomputation for memory is the actual idea.

    Think about
    -----------
    * Why is the log-sum-exp the *right* thing to save? What can you reconstruct
      from ``L`` plus the inputs, and how much memory did saving one scalar per
      row rather than one per score entry buy you?
    * With ``causal=True``, whole tiles are either fully masked or fully visible.
      Which tiles can be skipped outright, and what does that do to the FLOP count?
    * Where exactly does :func:`online_softmax_update` fit into your loop nest?
    """
    # TODO: loop over Q tiles (outer) and K/V tiles (inner), streaming the softmax
    raise NotImplementedError


def flash_attention_backward(
    dout: np.ndarray,
    cache: dict,
    block_q: int = 32,
    block_k: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tiled attention backward, recomputing scores on the fly.

    Parameters
    ----------
    dout : shape ``(batch, n_heads, seq_q, d_v)``
    cache : from :func:`flash_attention_forward`
    block_q, block_k : tile sizes

    Returns
    -------
    dQ, dK, dV : shapes matching the forward inputs

    Contract
    --------
    Must match :func:`scaled_dot_product_attention_backward` to ~1e-5, under the
    same "no full score matrix" restriction as the forward.

    Think about
    -----------
    * You did not save the attention weights. Given ``L`` from the forward, how do
      you recover a *tile* of weights exactly, without a second pass over the row?
    * The softmax-Jacobian term needs a per-row scalar that depends on the whole
      row of ``dout`` and ``output``. Look at what ``rowsum(dout * output)``
      computes and convince yourself it is exactly that scalar. Note that it is
      computable in one cheap elementwise pass **before** the tile loop — that
      precomputation is the key structural trick of the flash backward.
    * ``dQ`` accumulates across K tiles; ``dK``/``dV`` accumulate across Q tiles.
      One of those is awkward if you keep the forward's loop order. Which one, and
      what do real implementations do about it? (Consider a second, separately
      ordered loop nest.)
    * Why does this end up *faster* than the non-flash backward on a GPU despite
      doing strictly more arithmetic?
    """
    # TODO
    raise NotImplementedError


# =============================================================================
# Layer 10 — Deployment: quantisation
# =============================================================================

def quantize_symmetric_int8(x: np.ndarray, axis: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric absmax quantisation to int8.

    Parameters
    ----------
    x : float32 array
    axis : ``None`` → one scale for the whole tensor (per-tensor).
           An int → one scale per slice along that axis (per-channel).

    Returns
    -------
    q : int8 array, same shape as ``x``, values within ``[-127, 127]``
    scale : float32. Scalar if ``axis is None``, otherwise shaped so that
            ``q * scale`` broadcasts back against ``x``.

    Contract
    --------
    Use ``127`` as the positive limit, not ``128``, so the range stays symmetric.
    Round half away from zero, then clip. An all-zero tensor must not produce
    ``nan`` — guard the degenerate scale.

    Think about
    -----------
    * Why symmetric (no zero-point) for weights, but asymmetric for activations
      coming out of a ReLU?
    * Per-channel costs one extra float per output channel and typically recovers
      most of the accuracy lost by per-tensor. What property of trained weight
      matrices explains that?
    """
    # TODO
    raise NotImplementedError


def dequantize_int8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Map int8 back to float32. Left-inverse of the above, up to rounding error.

    Think about
    -----------
    * Bound the worst-case per-element error in terms of ``scale``. Then predict
      the round-trip error the test should see, before you run it.
    """
    # TODO
    raise NotImplementedError


def quantized_matmul_int8(
    a_q: np.ndarray,
    a_scale: np.ndarray,
    b_q: np.ndarray,
    b_scale: np.ndarray,
) -> np.ndarray:
    """Simulated int8 GEMM with int32 accumulation and a float32 rescale.

    Parameters
    ----------
    a_q : int8, shape ``(M, K)``
    a_scale : scale(s) for ``a``
    b_q : int8, shape ``(K, N)``
    b_scale : scale(s) for ``b``

    Returns
    -------
    out : float32, shape ``(M, N)``

    Contract
    --------
    Accumulate in **int32**, then rescale once at the end. Do not dequantise the
    inputs and call a float matmul — that would defeat the point and hide the
    overflow question below.

    Phase 2/3 note
    --------------
    This mirrors what tensor cores do: int8 multiply, int32 accumulate, scale in
    the epilogue. In Phase 3 that epilogue rescale is exactly where you would fuse
    bias and activation as well.

    Think about
    -----------
    * Worst case, how large can the int32 accumulator get for ``K = 4096`` with
      int8 inputs? Compare against ``2^31``. Is int32 accumulation actually safe,
      and would int16 have been?
    * With per-channel scales on both sides, is the final rescale still a single
      broadcast multiply? Work out the shapes.
    """
    # TODO
    raise NotImplementedError


# =============================================================================
# Quick manual test
# =============================================================================
if __name__ == "__main__":
    print("attention.py loaded successfully.")
    print("Start implementing from Layer 0 and work your way up.")
    print("See README.md for the recommended order and the full TODO list.")
    print("Run 'python -m unittest test_attention -v' after each implementation.")