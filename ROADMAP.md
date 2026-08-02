# ROADMAP

三个阶段的知识点清单。README 只放概览，细节在这里。

Phase 1 的每一项在 `attention.py` 里都有对应的桩函数和已写好的测试，
**进度以测试通过数为准**，这里的勾选只是给人看的索引。

## Phase 1 — NumPy

### Layer 0 — 基础工具

- [x] `softmax` — 数值稳定的 max 平移
- [x] `softmax_backward` — Jacobian 的对角 + 外积结构
- [x] `create_causal_mask` — 加性上三角 `-inf`
- [x] `create_padding_mask` — 变长 batch 对齐
- [ ] `dropout_forward` / `dropout_backward` — inverted dropout，为什么 eval 时不用缩放
- [ ] `create_sliding_window_mask` — Mistral 式局部窗口

### Layer 0b — 循环 / 分块内核（**Phase 2 的桥梁**）

- [ ] `matmul_naive` — 三重循环，禁用 `@`/`matmul`/`einsum`（测试会用 AST 检查）
- [ ] `matmul_tiled` — cache blocking，必须处理不能整除的边界 tile
- [ ] `online_softmax_update` — 流式 softmax 递推，FlashAttention 的内核

> 这三个是整个项目里最容易被跳过、但对 Phase 2/3 最关键的部分。
> `A @ B` 在 NumPy 里是一个字符，在 CUDA 里是两百行。不在这里把循环写出来，
> Phase 2 就不是"移植"而是"重写"。

### Layer 0c — 激活函数

- [ ] `gelu_forward` / `gelu_backward` — tanh 近似 vs 精确 erf 两条分支
- [ ] `silu_forward` / `silu_backward` — 注意大负数的 exp 溢出
- [ ] `swiglu_forward` / `swiglu_backward` — 为什么 LLaMA 的 `d_ff` 是 8/3·d_model

### Layer 1 — 线性层

- [ ] `Linear.forward` / `Linear.backward` — 高维输入要 flatten 到 2D
- [ ] `Embedding.forward` / `Embedding.backward` — **scatter-add**，重复 token 是关键；
      对应 CUDA 里的 `atomicAdd`

### Layer 2 — Scaled Dot-Product Attention

- [ ] `scaled_dot_product_attention_forward`
- [ ] `scaled_dot_product_attention_backward`

### Layer 3 — Multi-Head Attention

- [ ] `MultiHeadAttention.forward` — self / cross 两种用法
- [ ] `MultiHeadAttention.backward` — self-attention 时三条路径的梯度要相加
- [ ] `repeat_kv` / `repeat_kv_backward` — MQA / GQA，KV cache 才是瓶颈

### Layer 4 — 归一化

- [ ] `LayerNorm.forward` / `LayerNorm.backward`
- [ ] `RMSNorm.forward` / `RMSNorm.backward` — 少了哪一个修正项，为什么

### Layer 5 — Transformer Block

- [ ] `AttentionBlock.forward` / `AttentionBlock.backward` — pre-LN + 残差

### Layer 6 — 位置信息

- [ ] `sinusoidal_positional_encoding` — 原始 Transformer 的固定编码
- [ ] `rope_precompute_freqs` / `rope_apply` / `rope_backward` — split-half 约定，
      `offset` 参数是 KV cache 正确性的命门
- [ ] `alibi_slopes` / `alibi_bias` — 无参数、可外推的相对位置偏置

### Layer 7 — 训练

- [ ] `cross_entropy_forward` / `cross_entropy_backward` — 融合 log-sum-exp、
      `ignore_index`、label smoothing
- [ ] `clip_grad_norm` — 全局范数，不是逐张量
- [ ] `lr_cosine_with_warmup` — warmup + cosine，注意 off-by-one
- [ ] `SGD.step` — momentum / nesterov / weight decay
- [ ] `AdamW.step` — decoupled weight decay 到底解耦在哪一步

### Layer 8 — 推理

- [ ] `KVCache.append` / `KVCache.reset` — 预分配 + 长度游标，不要 concat
- [ ] `MultiHeadAttention.forward_incremental` — 逐 token 解码必须和整段 causal
      forward **数值一致**，这是全项目最有价值的一个测试
- [ ] `apply_temperature` / `top_k_filter` / `top_p_filter`
- [ ] `sample_from_logits`（逆 CDF 采样）/ `greedy_select`

### Layer 9 — FlashAttention（项目重心）

- [ ] `flash_attention_forward` — 分块流式 softmax，全程不得出现
      `(seq_q, seq_k)` 大小的数组，只保存每行的 log-sum-exp
- [ ] `flash_attention_backward` — 重算 + `rowsum(dO ∘ O)` 技巧；
      dQ 沿 K 块累加、dK/dV 沿 Q 块累加，循环序需要重新组织

> 同一个算法贯穿三个阶段：这里是 NumPy 的 tile 循环，Phase 2 变成真实的 C 循环，
> Phase 3 里 Q tile 成为 thread block、KV tile 进 shared memory、
> 流式 softmax 状态进寄存器。先在 NumPy 里把它写对。

### Layer 10 — 部署：量化

- [ ] `quantize_symmetric_int8` — per-tensor / per-channel，absmax，±127
- [ ] `dequantize_int8`
- [ ] `quantized_matmul_int8` — int8 乘、**int32 累加**、epilogue 里 rescale

### Phase 1 收尾（尚未留桩，建议按序补）

这些无法用单元测试逐个校验，需要独立脚本，但它们是"我真的会了"的唯一证据：

- [ ] **数值梯度检查**：中心差分对比解析梯度。Phase 2 没有 autograd，这将是唯一的 oracle，
      必须先在 NumPy 里建立起来
- [ ] **端到端玩具训练**：char-level 语言模型跑通 Embedding → Block×N → 输出头 → CE loss →
      AdamW，看 loss 是否真的下降。任何一处 backward 写错，loss 曲线都会暴露
- [ ] **golden tensor dump**：把各算子的输入/输出/梯度存成 `.npz`，
      作为 Phase 2/3 的跨语言回归基准。这套东西不建立，Phase 2 就只能靠肉眼调试
- [ ] 权重初始化：Xavier / He / 残差分支的 `1/sqrt(2N)` 缩放
- [ ] 梯度累积、gradient checkpointing（用重算换显存，和 FlashAttention 同一思想）
- [ ] 混合精度模拟：fp16 存储 + fp32 累加 + loss scaling
- [ ] Pre-LN vs Post-LN 的稳定性对比实验

## Phase 2 — C / C++

- [ ] 张量表示：shape / stride / 连续性，先决定是否支持非连续视图
- [ ] 从 `matmul_naive` 起步：三种循环序各写一遍，实测差多少倍
- [ ] `matmul_tiled`：分块 + 边界处理，对比不同 tile 大小
- [ ] 手写 backward，用 Phase 1 的 golden tensor 校验（**没有 autograd 兜底**）
- [ ] 内存布局实验：`(B,S,H,D)` vs `(B,H,S,D)`，量化 transpose 的代价
- [ ] SIMD（AVX2/NEON）与 OpenMP 多线程
- [ ] FlashAttention 的 C 版本
- [ ] Roofline 分析：算术强度、访存带宽，判断每个算子是 compute-bound 还是 memory-bound

## Phase 3 — CUDA

- [ ] elementwise 起步：GELU / RMSNorm / RoPE，练 grid-stride loop 与访存合并
- [ ] 归约：softmax 的 block 内归约、warp shuffle
- [ ] GEMM：naive → shared memory tiling → register tiling → double buffering
- [ ] FlashAttention 前向：Q tile → thread block，KV tile → shared memory
- [ ] FlashAttention 反向：重算策略与两套循环序
- [ ] 算子融合：QKV projection + RoPE 融合；bias + activation 融进 GEMM epilogue
- [ ] flash-decoding：KV 维度 split-K，解决 decode 阶段并行度不足
- [ ] Paged KV cache（vLLM 式 block table）
- [ ] int8 / fp8 tensor core 路径
- [ ] Nsight Compute profiling：占用率、访存效率、bank conflict
