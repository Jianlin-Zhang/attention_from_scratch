# Phase 3 — CUDA

未开始。建立在 Phase 2 之上：C 版本里想清楚的循环序、分块和内存布局，
在这里映射成 grid / block / shared memory / 寄存器。

知识点清单见 [../ROADMAP.md](../ROADMAP.md)。

注意本阶段需要 **CUDA 版 PyTorch** 来对比 kernel 结果，请**另建一个 conda 环境**，
不要改根目录的 `requirements.txt` —— 那里锁的是 CPU 版 torch 2.2.0，
一旦放宽，Phase 1 导出的 `golden/` 参考数值就不再可复现。
