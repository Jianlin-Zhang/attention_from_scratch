# 项目协作规则 — attention_from_scratch

> **这份文件与 `.kiro/steering/collaboration-rules.md` 是镜像**，内容必须保持一致。
> 前者给 Claude Code 用，后者给 Kiro CLI 自动加载。改一处记得改另一处。

## 项目目标

通过**纯手撕**实现 attention 的全部细节（训练 / 推理 / 部署三类知识点），建立底层认知。
不是为了产出一个可用的库，而是为了让作者本人真正掌握每一个算子的数学与实现。

三部曲：

1. **Phase 1 — NumPy**：纯 Python + NumPy 手撕，含 forward / backward，用 PyTorch 做 oracle 验证。
2. **Phase 2 — C/C++**：更底层重写，显式循环、显式内存管理、显式布局。
3. **Phase 3 — CUDA**：把 Phase 2 里的算子搬到 GPU，做 tiling / shared memory / warp 级优化。

## 对 AI 助手的硬性约束（最重要）

1. **永远不要替作者写实现代码。** 任何算子的函数体、任何数学公式的代码化，都必须由作者本人完成。
2. 作者来提问 / 讨论时，只能做**思路引导**：反问、拆解、给出验证思路、指出方向性错误、给出可读的参考资料。
   不给可直接粘贴的答案。
3. **不要在注释或文档里泄漏推导结果。** 尤其禁止在 `TODO` 里直接写出反向传播公式。
   允许写的内容：函数签名、输入输出 shape/dtype 契约、语义约定（如 mask 是加性的）、
   引导性问题（"这一步的 Jacobian 是稠密的还是对角的？"）、以及该算子在 Phase 2/3 的意义。
4. **允许 AI 直接写的东西**（这些是脚手架/规格，不是学习内容）：
   - 单元测试（`test_*.py`）—— 测试是契约与 oracle，作者已明确授权。
   - 函数签名 + docstring + `raise NotImplementedError` 的桩（stub）。
   - README、路线图、依赖与环境配置、构建脚本、CI 配置。
5. 如果作者直接要求"给我写这个算子的实现"，先提醒本规则第 1 条，再问他是想要**提示**还是想**跳过**这个知识点。

## 判定"某个知识点该不该加进 Phase 1"的标准

只有同时满足下面两条才值得在 NumPy 阶段手撕：

- 它是 attention 训练/推理/部署链路上的真实知识点；
- 它在 Phase 2（C）或 Phase 3（CUDA）里有明确对应物（显式循环、tile 划分、内存布局、原子加、数值精度等）。

反例：不要在 NumPy 阶段引入纯工程性的框架抽象（自动微分引擎、计算图、模块注册表），
那会偏离"手撕算子"的主线。

## 环境与命令

本项目有**专属 conda 环境 `attn-scratch`**，不要用其他项目的环境（历史上曾误用
`openvla` / `openvla-read`，跨机器名字不一致，已废弃这种做法）。

```bash
conda env create -f environment.yml     # 首次，在仓库根目录
conda activate attn-scratch

cd phase1_numpy
python -m unittest test_attention -v                     # 全量
python -m unittest test_attention.TestRoPE -v            # 单个知识点
```

版本固定在 `requirements.txt`：Python 3.11 / numpy 1.26.4 / torch 2.2.0 (**CPU**)。

依赖只有两个，且职责严格区分：

- `numpy` —— `attention.py` 唯一的运行时依赖；
- `torch` —— **只有测试用**，作为数值 oracle 提供参考值和参考梯度，
  绝不参与被学习的代码路径。CPU 版足够。

torch 2.2.0 的 API 边界：

- `F.scaled_dot_product_attention` 可用；
- `F.rms_norm` **不可用**（2.4 才加入），测试里已自写参考实现；
- `F.cross_entropy(..., label_smoothing=)`、`torch.optim.AdamW`、`clip_grad_norm_` 均可用。

Phase 3 需要 CUDA 版 torch 时，**另建一个环境**，不要放宽这个环境的固定版本，
否则 Phase 1 的参考数值就不再可复现。

## 目录结构约定

按**阶段**分目录，不按 src/tests 分 —— 三个阶段工具链完全不同，各自自成一个可构建单元。

```
phase1_numpy/     attention.py + test_attention.py（就两个文件，保持扁平，不要拆 src/tests）
phase2_c/         未开始
phase3_cuda/      未开始
golden/           跨阶段共享的参考张量：Phase 1 导出，Phase 2/3 回归比对
```

环境定义（`environment.yml` / `requirements.txt`）留在仓库根目录，
因为它同时也是跑 `golden/` 导出脚本的开发环境。

## 代码约定

- Batch-first，形状统一 `(batch, seq_len, features)`；多头张量为 `(batch, n_heads, seq, head_dim)`。
- mask 一律是**加性**的：`0` = 允许，`-inf` = 屏蔽，在 softmax 之前相加。
- forward 返回 `(output, cache)` 或把中间量存进 `self._cache`；backward 只消费 cache，不重算 forward
  （唯一例外是 FlashAttention backward，它故意重算以省显存 —— 这是知识点本身）。
- 参数梯度写在 `self.dW` / `self.db` / `self.dgamma` 之类的属性上，输入梯度作为返回值。
- 新增知识点必须同时补一个 unittest，先让它红，再动手实现。

## 进度记录方式

**进度就是测试通过数**（`python -m unittest test_attention`），不额外维护清单。
`ROADMAP.md` 是知识点索引，勾选只为方便人看；真实状态永远看测试。
commit message 里点名本次完成的知识点。
