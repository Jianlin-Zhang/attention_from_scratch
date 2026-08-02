# attention_from_scratch

从零手撕 attention 的全部细节 —— 训练、推理、部署三条链路上的知识点，一个都不放过。

目标不是造一个可用的库，而是**把每个算子的数学和实现都亲手写一遍**，建立底层认知。

## 三部曲

| 阶段 | 语言 | 关注点 | 状态 |
|------|------|--------|------|
| Phase 1 | Python + NumPy | 数学正确性、forward/backward，以 PyTorch 为 oracle | 进行中 |
| Phase 2 | C / C++ | 显式循环、内存布局、cache blocking，无 autograd | 未开始 |
| Phase 3 | CUDA | tiling、shared memory、warp 级归约、算子融合 | 未开始 |

每一层都要能对齐上一层的数值结果 —— Phase 1 的 NumPy 实现就是 Phase 2/3 的黄金参考。

完整知识点清单见 [ROADMAP.md](ROADMAP.md)。

## 环境

专属 conda 环境 `attn-scratch`，版本全部固定，跨机器可复现。

```bash
conda env create -f environment.yml     # 首次
conda activate attn-scratch

cd phase1_numpy
python -m unittest test_attention -v              # 全量
python -m unittest test_attention.TestRoPE -v     # 单个知识点
```

依赖只有两个，职责严格区分：`numpy` 是 `attention.py` 唯一的运行时依赖；
`torch`（2.2.0 **CPU** 版）**只给测试用**，作为数值 oracle 提供参考值与参考梯度，
绝不参与被学习的代码路径。Phase 3 要拿 GPU torch 对比 kernel 结果时另建环境，
不要放宽这里的版本锁，否则 Phase 1 的参考数值不再可复现。

## 进度

`attention.py` 里每个待实现的算子都是一个桩，对应测试已经写好，
所以**进度就是测试通过数**，不用另外维护清单：

```bash
cd phase1_numpy && python -m unittest test_attention 2>&1 | tail -3
```

当前 151 个用例通过 7 个 —— `softmax` / `softmax_backward` /
`create_causal_mask` / `create_padding_mask`。

工作流：**先跑测试让它红 → 推导 → 实现 → 跑到绿 → commit**。

## 目录

按**阶段**分目录，而不是按 src/tests 分 —— 三个阶段的工具链完全不同
（`unittest` / `make`+`gcc` / `nvcc`），各自自成一个可构建单元。

```
phase1_numpy/           Phase 1，就两个文件，保持扁平
  attention.py            全部实现（含桩函数）；docstring 只给形状契约和引导问题，不给公式
  test_attention.py       单元测试，以 PyTorch 或自一致性为 oracle
phase2_c/               Phase 2，未开始
phase3_cuda/            Phase 3，未开始
golden/                 跨阶段共享的参考张量，Phase 1 导出、Phase 2/3 回归比对

ROADMAP.md              三个阶段的知识点清单
requirements.txt        Phase 1 依赖固定版本（唯一来源）
environment.yml         conda 环境定义，引用 requirements.txt
CLAUDE.md               协作规则：AI 只引导，不代写任何实现
.kiro/steering/collaboration-rules.md   同上，Kiro CLI 自动加载 —— 两者是镜像
```

环境定义留在根目录：它同时也是跑 `golden/` 导出脚本的开发环境。
Phase 3 需要 CUDA 版 torch 时另建环境，不要动这里。
