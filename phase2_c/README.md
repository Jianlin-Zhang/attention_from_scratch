# Phase 2 — C / C++

未开始。**先把 Phase 1 做完**，尤其是 `matmul_naive` / `matmul_tiled` /
`online_softmax_update` 这三个显式循环的算子，以及 `golden/` 里的参考张量 ——
它们是本阶段唯一的正确性来源（这里没有 autograd 兜底）。

知识点清单见 [../ROADMAP.md](../ROADMAP.md)。

建议目录（真正动手时再建，别提前搭空架子）：

```
src/        算子实现
include/    头文件
tests/      比对 ../golden/*.npz 的回归测试
Makefile
```
