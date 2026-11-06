# SqueezeNet v1.0 CIFAR-10 training baselines (PyTorch / TensorFlow)

Batch 32, float32, Adam lr 5e-4, loss = NLL + 5e-5*sum(kernel^2); one epoch = 250 steps (8000 rows of CIFAR-10 data_batch_1).  `aug` = on-the-fly pad-4/random-crop/h-flip on the batch inside the timed step; `off` = pre-scaled batches straight from the dataset.  Timing: 10 warmup steps excluded, then >=100 steps and >=20 s.  NLL = mean sparse softmax cross-entropy of the last timed step (chance = ln 10 = 2.303).

> **Caveat (2026-10-30 run):** every `cpu` row below was measured between 14:38 and 14:59 local time while the main session was rebuilding the library and running its own 16-thread benchmarks on the same 16-core/32-thread machine, so the CPU rows are contaminated (lower bounds; the 32-thread collapse is at least partly contention). Re-run them on a quiet machine with `benchmarks/frameworks/run_all.sh torch-cpu tf-cpu extra` (commands in `benchmarks/frameworks/README.md`). The `cuda` rows (14:51-14:56) ran with the GPU otherwise idle and are the numbers to keep. `tensorflow_cpu_8_warmup_cosine.json` was not produced (run stopped on request).

| framework | device | threads | aug | steps/s | img/s | s/epoch | timed steps | CPU util | final NLL | notes |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| torch 2.11.0+cu128 | cpu | 1 | on | 10.79 | 345 | 23.2 | 216 | 1.0x | 2.303 | NLL stuck at ln10 (dead head?) |
| torch 2.11.0+cu128 | cpu | 1 | on | 13.50 | 432 | 18.5 | 271 | 1.0x | 2.303 | channels_last, NLL stuck at ln10 (dead head?) |
| torch 2.11.0+cu128 | cpu | 1 | off | 10.85 | 347 | 23.0 | 218 | 1.0x | 2.305 |  |
| torch 2.11.0+cu128 | cpu | 4 | on | 26.53 | 849 | 9.4 | 531 | 4.0x | 2.303 | NLL stuck at ln10 (dead head?) |
| torch 2.11.0+cu128 | cpu | 4 | on | 27.10 | 867 | 9.2 | 542 | 4.0x | 2.303 | channels_last, NLL stuck at ln10 (dead head?) |
| torch 2.11.0+cu128 | cpu | 4 | off | 25.10 | 803 | 10.0 | 503 | 4.0x | 2.266 |  |
| torch 2.11.0+cu128 | cpu | 8 | on | 31.58 | 1011 | 7.9 | 632 | 8.0x | 2.303 | NLL stuck at ln10 (dead head?) |
| torch 2.11.0+cu128 | cpu | 8 | on | 28.67 | 917 | 8.7 | 574 | 8.0x | 2.303 | channels_last, NLL stuck at ln10 (dead head?) |
| torch 2.11.0+cu128 | cpu | 8 | off | 30.93 | 990 | 8.1 | 619 | 8.0x | 2.199 |  |
| torch 2.11.0+cu128 | cpu | 8 | on | 29.87 | 956 | 8.4 | 598 | 8.0x | 2.303 | NLL stuck at ln10 (dead head?) |
| torch 2.11.0+cu128 | cpu | 8 | on | 28.49 | 912 | 8.8 | 600 | 8.0x | 1.725 | warmup-cosine schedule (training sanity run) |
| torch 2.11.0+cu128 | cpu | 16 | on | 28.01 | 896 | 8.9 | 561 | 16.0x | 2.303 | NLL stuck at ln10 (dead head?) |
| torch 2.11.0+cu128 | cpu | 16 | on | 32.09 | 1027 | 7.8 | 642 | 15.9x | 2.303 | channels_last, NLL stuck at ln10 (dead head?) |
| torch 2.11.0+cu128 | cpu | 16 | off | 28.96 | 927 | 8.6 | 580 | 15.9x | 2.303 | NLL stuck at ln10 (dead head?) |
| torch 2.11.0+cu128 | cpu | 32 | on | 9.74 | 312 | 25.7 | 195 | 30.9x | 2.303 | NLL stuck at ln10 (dead head?) |
| torch 2.11.0+cu128 | cpu | 32 | on | 1.93 | 62 | 129.7 | 100 | 30.5x | 2.303 | channels_last, NLL stuck at ln10 (dead head?) |
| torch 2.11.0+cu128 | cpu | 32 | off | 7.73 | 247 | 32.4 | 168 | 30.8x | 2.303 | NLL stuck at ln10 (dead head?) |
| torch 2.11.0+cu128 | cuda (RTX 5080 Laptop GPU) | - | on | 223.86 | 7164 | 1.1 | 4478 | 1.0x | 2.057 | tf32=default, native sm |
| torch 2.11.0+cu128 | cuda (RTX 5080 Laptop GPU) | - | on | 264.53 | 8465 | 0.9 | 5291 | 1.0x | 2.303 | channels_last, tf32=default, native sm, NLL stuck at ln10 (dead head?) |
| torch 2.11.0+cu128 | cuda (RTX 5080 Laptop GPU) | - | on | 282.89 | 9052 | 0.9 | 5658 | 1.0x | 1.770 | tf32=default, cudnn.benchmark, native sm |
| torch 2.11.0+cu128 | cuda (RTX 5080 Laptop GPU) | - | off | 298.46 | 9551 | 0.8 | 5970 | 1.0x | 0.589 | tf32=default, cudnn.benchmark, native sm |
| torch 2.11.0+cu128 | cuda (RTX 5080 Laptop GPU) | - | on | 306.97 | 9823 | 0.8 | 1500 | 1.0x | 1.666 | tf32=default, native sm, lr=0.0001 |
| torch 2.11.0+cu128 | cuda (RTX 5080 Laptop GPU) | - | off | 279.61 | 8948 | 0.9 | 5593 | 1.0x | 0.610 | tf32=default, native sm |
| torch 2.11.0+cu128 | cuda (RTX 5080 Laptop GPU) | - | on | 246.83 | 7899 | 1.0 | 4937 | 1.0x | 1.290 | tf32=off, native sm |
| torch 2.11.0+cu128 | cuda (RTX 5080 Laptop GPU) | - | on | 290.31 | 9290 | 0.9 | 2490 | 1.0x | 1.463 | tf32=default, native sm, warmup-cosine schedule (training sanity run) |
| tf 2.21.0 | cpu | 1 | on | 25.07 | 802 | 10.0 | 502 | 1.5x | 2.309 | mode=graph, inter-op=2 |
| tf 2.21.0 | cpu | 1 | on | 15.46 | 495 | 16.2 | 310 | 1.0x | 2.116 | mode=graph, inter-op=1 |
| tf 2.21.0 | cpu | 1 | off | 24.78 | 793 | 10.1 | 496 | 1.5x | 2.176 | mode=graph, inter-op=2 |
| tf 2.21.0 | cpu | 4 | on | 42.77 | 1368 | 5.8 | 856 | 3.7x | 2.346 | mode=graph, inter-op=2 |
| tf 2.21.0 | cpu | 4 | on | 32.35 | 1035 | 7.7 | 647 | 3.2x | 2.104 | mode=graph, inter-op=1 |
| tf 2.21.0 | cpu | 4 | off | 44.26 | 1416 | 5.6 | 886 | 3.7x | 2.302 | mode=graph, inter-op=2, NLL stuck at ln10 (dead head?) |
| tf 2.21.0 | cpu | 8 | on | 47.57 | 1522 | 5.3 | 952 | 6.2x | 2.303 | mode=graph, inter-op=2, NLL stuck at ln10 (dead head?) |
| tf 2.21.0 | cpu | 8 | on | 9.64 | 308 | 25.9 | 193 | 2.4x | 2.294 | mode=eager, inter-op=2 |
| tf 2.21.0 | cpu | 8 | on | 35.53 | 1137 | 7.0 | 711 | 4.8x | 2.303 | mode=graph, inter-op=1, NLL stuck at ln10 (dead head?) |
| tf 2.21.0 | cpu | 8 | off | 47.48 | 1519 | 5.3 | 950 | 6.2x | 2.007 | mode=graph, inter-op=2 |
| tf 2.21.0 | cpu | 8 | on | 45.87 | 1468 | 5.4 | 918 | 6.2x | 2.303 | mode=graph, inter-op=2, NLL stuck at ln10 (dead head?) |
| tf 2.21.0 | cpu | 8 | on | 16.48 | 527 | 15.2 | 330 | 5.7x | 2.304 | mode=xla, inter-op=2, NLL stuck at ln10 (dead head?) |
| tf 2.21.0 | cpu | 16 | on | 43.79 | 1401 | 5.7 | 876 | 11.2x | 1.997 | mode=graph, inter-op=2 |
| tf 2.21.0 | cpu | 16 | off | 42.60 | 1363 | 5.9 | 852 | 11.0x | 2.162 | mode=graph, inter-op=2 |
| tf 2.21.0 | cpu | 32 | on | 26.46 | 847 | 9.4 | 530 | 10.7x | 2.299 | mode=graph, inter-op=2 |
| tf 2.21.0 | cpu | 32 | off | 37.52 | 1201 | 6.7 | 751 | 16.7x | 1.797 | mode=graph, inter-op=2 |
| tf 2.21.0 | cuda (RTX 5080 Laptop GPU) | - | on | 177.98 | 5695 | 1.4 | 3560 | 1.1x | 2.302 | mode=graph, NLL stuck at ln10 (dead head?) |
| tf 2.21.0 | cuda (RTX 5080 Laptop GPU) | - | off | 184.16 | 5893 | 1.4 | 3684 | 1.0x | 2.303 | mode=graph, NLL stuck at ln10 (dead head?) |
| tf 2.21.0 | cuda (RTX 5080 Laptop GPU) | - | on | 189.30 | 6058 | 1.3 | 2490 | 1.2x | 1.612 | mode=graph, warmup-cosine schedule (training sanity run) |
| tf 2.21.0 | cuda (RTX 5080 Laptop GPU) | - | on | 366.21 | 11719 | 0.7 | 7325 | 1.0x | 1.072 | mode=xla |

Source records: `pytorch_cpu_1.json`, `pytorch_cpu_1_channels_last.json`, `pytorch_cpu_1_noaug.json`, `pytorch_cpu_4.json`, `pytorch_cpu_4_channels_last.json`, `pytorch_cpu_4_noaug.json`, `pytorch_cpu_8.json`, `pytorch_cpu_8_channels_last.json`, `pytorch_cpu_8_noaug.json`, `pytorch_cpu_8_rep2.json`, `pytorch_cpu_8_warmup_cosine.json`, `pytorch_cpu_16.json`, `pytorch_cpu_16_channels_last.json`, `pytorch_cpu_16_noaug.json`, `pytorch_cpu_32.json`, `pytorch_cpu_32_channels_last.json`, `pytorch_cpu_32_noaug.json`, `pytorch_cuda_0.json`, `pytorch_cuda_0_channels_last.json`, `pytorch_cuda_0_cudnnbench.json`, `pytorch_cuda_0_cudnnbench_noaug.json`, `pytorch_cuda_0_lr1e-4.json`, `pytorch_cuda_0_noaug.json`, `pytorch_cuda_0_tf32off.json`, `pytorch_cuda_0_warmup_cosine_10ep.json`, `tensorflow_cpu_1.json`, `tensorflow_cpu_1_inter1.json`, `tensorflow_cpu_1_noaug.json`, `tensorflow_cpu_4.json`, `tensorflow_cpu_4_inter1.json`, `tensorflow_cpu_4_noaug.json`, `tensorflow_cpu_8.json`, `tensorflow_cpu_8_eager.json`, `tensorflow_cpu_8_inter1.json`, `tensorflow_cpu_8_noaug.json`, `tensorflow_cpu_8_rep2.json`, `tensorflow_cpu_8_xla.json`, `tensorflow_cpu_16.json`, `tensorflow_cpu_16_noaug.json`, `tensorflow_cpu_32.json`, `tensorflow_cpu_32_noaug.json`, `tensorflow_cuda_0.json`, `tensorflow_cuda_0_noaug.json`, `tensorflow_cuda_0_warmup_cosine_10ep.json`, `tensorflow_cuda_0_xla.json`
