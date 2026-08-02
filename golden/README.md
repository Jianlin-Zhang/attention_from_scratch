# golden/ — 跨阶段参考张量

三个阶段之间唯一的共享契约，不属于任何单一阶段。

Phase 1 的 NumPy 实现（已用 PyTorch 校验过）把每个算子的输入、输出、梯度导出成
`.npz`，Phase 2 的 C 实现和 Phase 3 的 CUDA kernel 读取同一批文件做回归比对。

**为什么必须有这个目录**：Phase 2/3 没有 autograd，也不方便链 PyTorch。
没有黄金张量，C 和 CUDA 的 backward 就只能靠肉眼调试。

约定（等 Phase 1 收尾时落地，见 [../ROADMAP.md](../ROADMAP.md)）：

- 一个算子一个 `.npz`，文件名即算子名，例如 `flash_attention_forward.npz`
- 固定随机种子，形状取小值（能人肉核对），但要覆盖不能整除 tile 的边界情况
- 同时存 forward 的输入/输出和 backward 的 `dout` / `dQ,dK,dV`
- 一律 float32，行主序连续 —— C 和 CUDA 侧才好直接 `fread`
- 文件小，直接入库；它们是参考基准，不是可再生的中间产物
