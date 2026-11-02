# yue-metal — YuE2-3B 原生 Metal 推理引擎

English documentation: [README.en.md](README.en.md)

YuE2-3B 歌曲生成模型（AR 语义 token + NAR 声学流匹配 + Oobleck VAE）的原生推理引擎，
纯 C11 + Objective-C/Metal，面向 Apple Silicon（M 系统一体内存）。数值约定逐点复刻
参考栈 lyra（MLX 移植版，`/Volumes/Work/mlx-Yue`，含其 vendor 的官方
yue2 协议层），AR 贪心输出与 lyra 逐位一致（见「数值口径」）。

同机同请求对比（2026-11-02，442 字歌词 / cot=off / cfgScale=1.2 / seed=831001，
bf16 对 bf16，双方实测；采样轨迹不同属设计内——初始噪声为引擎自有流）：

| 引擎 | semantic | flow | save | e2e | RTF |
|---|---|---|---|---|---|
| lyra (MLX bf16) | 75.5s / 4995 tok（15.1 ms/token） | 106.5s | ≈2s | ≈184s | 0.92 |
| yue-metal（M15，CFG batch fusion） | 55.6s / 4730 tok（11.7 ms/token） | 84.8s | 2.6s | **143.5s** | **0.758** |

**整体快 ~1.2×**（e2e −22%，RTF 0.92 → 0.758）：分项 semantic **−22%**、
flow **−20%**。M15 的 batch fusion 把双分支
GEMM 的权重流从两遍并成一遍（见性能节）；flow 侧 yue 的 NAR/VAE 本就反超
（M7 起）。同请求双发 yue：RTF 0.725/0.763（采样轨迹不同）。

性能历程（semantic ms/token 为请求画像间不可直接比的量级参考，精确对比以
各条目的同请求 A/B 为准；RTF 同理）：

| 里程碑 | semantic（画像不同，仅看趋势） | e2e RTF | 主要手段 |
|---|---|---|---|
| M6 首版全管线 | — | 2.2 | 功能优先 |
| M7（四轮） | 74 → 28.6 | 1.15 | 层内 submit 合并、MPS linear、整步命令链、snake vForce、qkv 融合 GEMV |
| M8 | 28.6 | 0.97 | VAE 解码 GPU 化（save 6.2s → 0.6s） |
| M10 | 19.9 | 0.80 | decode split-K SDPA（1.91×，flash2 的 64 行 padding 税） |
| M11 | ≈18.7 | — | attention prologue 融合（hn+rope 单 dispatch，−7%） |
| M14 | 17.2-20.2 | ≈0.95 | decode GEMV 默认切 2D 块内核（−4.3%）+ 宿主尾归因修正 |
| **M15** | **10.3-12.6** | **0.758** | **CFG batch fusion（双分支单流，−40%）** |

累计：semantic **≈6.3×**、e2e RTF **≈2.9×**。调度侧（M12）、GEMM 内核
（M13）、编码/宿主（M14）已封盘——每一步的裁决与数据见性能节与
`tests/bench_dispatch.c` 头注释。

历史版本：M10（321 字请求画像）semantic 19.9 ms/token、RTF 0.80（lyra
同口径 ≈17.1/0.73）；M7 四轮 34.2s（RTF 1.15）→ M8 VAE GPU 化 28.8s
（RTF 0.97）。

## 模型与依赖

- 权重：`/Volumes/Work/models/YuE2.models/lyra-YuE2-3B`（ar-bf16 + nar-bf16 + qwen.tiktoken）
  与同级独立目录 `lyra-YuE2-3B-vae`（`--vae` 缺省时自动按 `<model>-vae` 推导）。
- 参考栈：`/Volumes/Work/mlx-Yue`（lyra MLX 包 + 其 `.venv`，仅测试 fixture 生成时使用）。
- 运行依赖：Metal、MetalPerformanceShaders(+Graph)、Accelerate、ffmpeg（FLAC 转码，PATH 中可用）。

## 构建

```sh
make yue            # 构建 ./yue（CLI，serve + repl 两种模式）
make m0-tests tokenizer-tests gemv-tests ar-tests sampling-tests nar-tests vae-tests
```

## 用法

```sh
# JSONL 服务模式
printf '%s\n' '{"id":"t","style":"upbeat pop","lyrics":"[verse]\nla la la\n[chorus]\nna na na","cot":"off","cfgScale":1.2,"seed":831001,"output":"/tmp/t.flac"}' \
  | ./yue serve -d /Volumes/Work/models/YuE2.models/lyra-YuE2-3B

# 常驻 REPL（linenoise）
./yue repl -d MODEL_DIR [--vae VAE_DIR]
```

REPL 指令：

```
/style TEXT        风格标签（必填）
/lyrics TEXT       歌词；\n 转义换行（必填）
/lyrics-file PATH  从 UTF-8 文件读歌词
/cot full|melody|off  解码模式（默认 full）
/cfg X             显式 CFG；0 = 自动
/seed N            基础种子（默认 831001，lyra 默认值；每次 /gen 后自增）
/out PREFIX        输出路径前缀（默认 out，产出 PREFIX-<n>.flac）
/show /gen /help /quit
```

## JSONL 协议（serve）

- 请求：`{"id","style","lyrics","cot","cfgScale","seed","output"}`（行上限 512KB；
  校验对齐 worker.py：cfgScale 仅 NUMBER 且 >0 生效、范围 [0,20]；seed ∈ [0,2^63)；
  output 必须带 `.flac`）。
- 事件：`state loading/ready(loadSeconds)` → `progress(id/stage/percent/elapsed[,
  heartbeat])` → `done(path/seconds/tokens/sampleRate)` 或 `error(id/message)`。
- 进度阶段：`abc 0.02-0.38`（cot=off 时空转）→ `semantic 0.40-0.78` → `flow 0.80-0.94`
  （心跳线程外推）→ `save 0.96-1.00`；4s 心跳不刷新宿主空闲计时。
- 取消 = 宿主杀进程组；stdin EOF 退出。
- GPU 单实例锁：`flock /tmp/yue-metal-engine-<uid>.lock`。yue2-web 现同为原生
  引擎（直接拉起本二进制的 serve 模式），双方共用此锁，天然互斥。
- NAR / VAE 惰性创建：ready 只加载 AR + tokenizer + 生成配置（~0.7s）。

## 性能（M11 后）

- **M15：CFG batch fusion——双分支合并单流，semantic −40%，反超 lyra**
  （2026-11-02）。probe H（`./tests/bench_dispatch H`，含 rows=2 vs 2×rows=1
  的 0 位差数值门）证明 MPS linear rows=2 与 rows=1 几乎同时间（fc1/down/lm
  batch-eff 0.94-1.00×，纯带宽主导）——双分支各读一遍权重（~12.6ms/token
  GEMM）是纯浪费。落地：`yue_ar_forward_open2`（[2,hidden] 单流 batch
  forward，两分支各行独立 attend/写自己的 cache，rope 表逐行携带各自相位——
  负前缀更长所以两 cache offset 不同）+ `qw_gpu_sdpa_decode_split_bf16_off`
  （query/output 元素偏移绑定变体，内核不变）+ `decode_step2`（[2,width]
  logits 一次投影+readback）。**关键语义：step 0 采样的是 prefill logits
  （pos.host/neg.host），host2 由每个 batched step 为下一 token 写入**——
  首版把 gen_branch.host 指进 host2 导致 step0 读未初始化内存（margin
  1e36）且清理双 free，已修（branch.host 保持自有 malloc，guided 按
  batch_on && step>0 分流）。serve（442 字歌词/cot=off/cfg1.2）：semantic
  **10.3→12.6 ms/token**（M14 默认 17.2→20.2，−40%；分解 enc
  3.1[MPS encode 84 次]+wait 7.1→9.5+宿主 0.12，与预测吻合），e2e RTF
  **0.725-0.763 vs lyra 0.92——反超**；semantic ~11.7 vs lyra ≈15.1
  （−22%，同请求口径）。**数值门**：melody_abc（abc 相位无 CFG 单分支）与 lyra 逐位
  保持；semantic 轨迹整体重排（rows=2 GEMM 归约序不同，分布等价），flip
  门内（off step12 margin0.0625、melody step8）；七套回归全绿（单流路径
  fixture 不受影响）。`YUE_CFG_BATCH=0` 回退串行双分支；debug：
  `YUE_BATCH_DBG=1` 逐步扫描 logits 非有限值并现场 dump
  /tmp/batch_nan_s*.bin。修复版 27 连发 serve（多 seed/同进程多请求）零
  失败。
- **M14：宿主尾归因修正 + decode GEMV 默认切 2D 块内核**（2026-11-02）。新增
  `YUE_HOST_TRACE=1`（每 512 步打印 semantic 每 token 的
  wall/enc/wait/read/guided/unpack/dist/sample 分解）。**归因修正**：semantic
  每 token ~19ms = CPU 编码 4.4（其中 MPSGraph linear encode ~3.1，MPSGraph
  固有税：命令复用/TensorData 缓存/graph 缓存均已默认开启）+ GPU 链 wait
  ~14.5（随 KV 增长）+ **宿主计算仅 0.11**——此前「宿主尾 2.7ms」是
  wall−busy 差值法把 flow 的 GPU 时间错摊给 semantic，作废。**enc 侧三条
  路全部侦察并止损**：per-column GEMV（enc −3.1 但 wait +6，死）、TensorOps
  GEMV `qw_tops_gemv_bf16`（matmul2d 直派发，0 位差 vs MPS，但 M=1 tile
  执行 −20~25% 带宽；TM=4 在本 MPP 编不过，TN 扫描无解，死；内核与 probe G
  TOPS 路径保留）、MPS encode 本身（无杠杆）。**落地：decode GEMV 默认切
  `qw_gemv2d_m64u8`**（main.c `default_gemv_env`，setenv overwrite=0，
  显式 env 仍赢；共享 vendored h3 的 qwen 默认不受影响）——2D 执行与 MPS
  同带而编码便宜（1.3 vs 4.4ms/token），serve 同窗口对比（同窗口=同 KV
  长度，KV 只依赖 step）**wall −0.84ms/token（−4.3%）**；七套回归全绿、
  melody_abc 与 lyra 逐位保持，semantic flip 漂移门内（off step24 margin0、
  melody step43 margin0）。M9「2D 生产 e2e 持平」的旧证被解释：flow 噪声
  掩盖了 ~0.45s 的 semantic 收益。semantic ~18.6ms/token（长请求口径，
  M15 后被 batch 取代，见 M15 条）。
- **M13：GEMM 赛道终审——满风扇重测 + 分档带宽校准，赛道关闭**（2026-11-02）。
  按 M12 纪律执行：风扇开满、probe 与生产同热态。①probe F 复跑与 M9 完全同带
  （all-MPS [1x] 12.43 vs 12.19；shipping 12.61；all-2D 12.16；all-GEMV(u8)
  13.35）——**probe 侧「~246GB/s 地板」不是热降频假象**；probe T 的 prod 变体
  17.12 vs M12 记录 18.44 的差才是热态噪声（F 跑完 15 分钟后热透时更快）。
  ②**M12 归因修正**：18.44 是 CFG 双分支总量，「GEMM ~12ms @246GB/s」把双分支
  当单分支——单分支 GEMM 实为 ~6ms @ ~470GB/s 有效带宽。③新增
  `./tests/bench_dispatch G`（真依赖单形状链 + 桥自环消除法）：**大形状
  fc1(100-107µs @471-503GB/s)、down(46-52µs @485-546)、lm(~259µs @514-519)
  已跑在流式上限（~520GB/s）的 90-100%**（占每层权重流量 75%）；小形状
  （q/k/v/o/qkv）probe 里的 612-2375GB/s 是 L2 复用假象（4 权重轮换
  16-34MB 驻留 L2；生产链每层被 fc1/down 冲刷——这正是 M9「probe 全胜、
  生产全平」之谜的下半场）。④合成闭环：生产每分支 GEMM 5.8-6.1ms vs 全形状
  500GB/s 理论地板 ~5.9ms——**GEMM 已在地板**；serve 满风扇 gpu busy
  97.9s/4850tok = 8.6ms/branch-token，分解 GEMM 6 + split SDPA 1.6 + 小内核
  0.6 + 派发 0.3 完全自洽。⑤**semantic 与 lyra 的差距不是 GEMM 效率**
  （lyra 口径 ~394GB/s 跑同权重需 ~7.2ms/分支，比我们慢）；差距在 split
  SDPA 随 KV 增长的成本与每 token 宿主尾（logits 回读+采样）。剩余 GEMM
  上限 ≤0.3ms/分支（semantic ~3%），且要求在真实 L2 压力下赢下小形状——
  即 M9 生产 A/B 两次输掉的同一场比赛。**裁决：GEMM 赛道关闭，无第四次**。
- **M11：attention prologue 融合落地为默认**（2026-11-01）。`qw_head_rms_rope_bf16`
  单 dispatch 覆盖 q+k 的 per-head RMSNorm + rotate_half RoPE（每层 4 dispatch
  → 1；每 (row,head) 一个 head_dim 线程 threadgroup，平方和保持参考实现的
  单线程升序 fma 链 → 与四内核路径逐位一致，位级 A/B rows=1/4 双形状 0/8 缓冲
  差异；norm/rope 相位全 threadgroup 并行）。**probe T [cfg2]：full 24.15 →
  fused 22.76 ms/branch（+1.39，吃掉 no-hn+rope ablation 上限 1.74 的 80%）**。
  第一版「每 head 单线程串行做 norm+rope」反而慢 0.45 ms——并行相位才是收益
  本体。serve A/B（同请求两发取二、按帧归一）：semantic 20.30 → **18.9 ms/frame**
  （−6.9%，复跑 18.96/18.86；GPU busy 21.08 → 19.74 ms/frame，收益纯 GPU 侧，
  per-frame submissions 持平）。`YUE_HN_ROPE_SEPARATE=1` 回退四内核路径。
  **rms 纵向融合（add+rms 合一）经 probe 判定为结构性零收益、未进生产**：
  add 派发本身 ~1.1µs，而残差加折进 256 线程 rms 关键路径的代价恰好抵消
  （probe −0.05 ms/branch，噪声内）；rms 归约序被位级锁死，无法换更快的
  归约树——该项与 melody_abc 硬门绑定，不可绕过。
- **M10：decode 步 split-K SDPA 落地为默认**（2026-11-01）。flash2 TensorOps
  tile 内核在 q_rows=1 时付满 64 行 QT padding（probe U：kv=513 134µs、kv=2048
  470µs）；`qw_sdpa_decode_split_bf16`+`qw_sdpa_decode_merge_bf16` 标量对读每行
  K/V 一次（28.5µs @513、50µs @2048，q 行 threadgroup 暂存）。第一版把单线程
  q 元素提升到行循环外，分数变成 q[tid]·Σk 而非全点积——只有 q 为常量时与真值
  逐位恒等，而 probe U 的 kv=1 mini-case 恰好把共享 q 张量覆写成常量 0.25，
  把 bug 伪装成「kv=1/64 三方逐位一致」；生产链 ar 门 1.25 relL2 / melody flip
  由此而来。修复后 ar 门 0.027，probe T [cfg2]：full 24.03 → sdpa→split
  18.45 ms/branch（sdpa 份额 8.85→5.58）。serve A/B（同请求、各两次，暖机后测）：
  semantic 38.1 → **19.9 ms/token**（1.91×），e2e 99.8 → 56.1s，RTF 1.24 →
  0.80；per-token GPU busy 48.5 → 31.2ms，per-token submissions 56.1/56.2 持平
  （命令链节奏不变，纯 GPU 侧收益）。`YUE_SDPA_FLASH2=1` 回退 flash2。
- semantic 双分支 28.6ms/token（14.3ms/branch-token，M8 口径、旧请求画像）；
  [gpu] encode 3.1s / wait 24.4s / busy 27.2s / submissions 43K；内存 live
  ~7.7GB、footprint ~9.6GB。
- **M8：VAE 解码 GPU 化**（save 6.2s → 0.6s，~10×；e2e 34.2 → 28.8s，RTF
  1.15 → 0.97）。Oobleck 解码器平移到 TensorOps matmul2d：Conv1d = 零 ring +
  每 tap 纯指针偏移 GEMM（dilation 作用于 taps 之间，固定 tap 的输入窗口是
  稠密视图，无需 gather）；ConvT = 每 tap GEMM + strided `+=` 写回；权重加载时
  GPU 重排 `[cout][k][cin] → [k][cout_pad][cin]`。tile 结构（core 256/halo 16）
  与三缓冲轮换不变；`YUE_VAE_CPU=1` 回退 Accelerate 参考路径。GPU fixture 对拍
  full40 worst 5.4e-06、small 3.7e-4（8 帧短序列本征带），音频门 2e-4 全过。
- M7 优化历程（同请求）：semantic 74 → 32.6 → 29.0 → 28.6 ms/token（累计 2.6×）。
  主要手段：①每层 4-5 次 submit 合并为 1 次（continue 流水线）；②MPS linear
  门槛放开启用 MPS GEMV；③decode 每 token 8 次 submit-wait → 1 次（整步命令链：
  双分支前向 + 两个 logits 投影编进同一条链，`yue_ar_forward_open`/
  `yue_ar_project_open` 链式 API）；④snake vForce；⑤q/k/v 融合 GEMV 默认启用
  （3 dispatch → 1，4096 列单 dispatch，位级一致）。
- **GEMV 路线终局否决（M8 谜团收口，勿再尝试）**。旧「probe 胜 3-4×/生产输
  16%」的矛盾 = 三重假象叠加：①probe B/D 链是假依赖（每 dispatch 同输入，
  相邻 kernel 可重叠），带宽全虚高；②probe 缺层尾 `continue` 时 MPSGraph
  encode 完全串行、数字双向虚假；③生产 GEMV A/B 从未设 `H3_GEMV_KERNEL`，
  全吃默认慢变体（unroll-2，~230GB/s）。修正后（`tests/bench_dispatch.c`
  probe F：真依赖链 + 层尾 continue + CFG 双分支 + 2048 步热态）probe 与生产
  方向一致：串行真依赖链下 MPS 12.2 vs GEMV-u8 13.3 ms/branch，serve A/B
  14.3 vs 17.4 ms/branch-token——**GEMV 全开仍输 MPS**。已证伪：mmap 零拷贝
  无关、CFG 双分支零惩罚（精确 2×）、logits 回读排空无关。per-column GEMV
  串行链天花板 ~222GB/s vs MPS ~246GB/s，理论收益上限仅 ~1.1×；lyra 的
  ~394GB/s 证明 MLX 式 2D 块 + simdgroup 归约 kernel 存在空间，属未来内核
  工程（先过 probe F 再进生产 A/B）。
- **MLX 式 2D 块 GEMV 重写已完成并二次否决（2026-11-01，GEMV 路线终局关闭）**。
  `qw_gemv2d_*`（SM=1/SN=32 整 simd 走一行 K 滑块、每 thread TM 行独立 fma 链、
  TN=8 uint4 载荷、BN=1 纯 simd 树归约零 barrier）+ 融合 `qw_gemv2d_qkv_*`
  （单 dispatch q/k/v，4096 行统一网格，与三连 2D dispatch 位级一致）。
  test_gemv A-E 全绿（e_k 映射 ×16 变体、行/尾部映射含 K-tail 2056 与行-tail
  300、宿主 fp32 容差对拍、融合 2D qkv 位级一致）。probe S：小矩阵灾难确实被
  修复（k/v 63→471GB/s、q/o 261→735GB/s，假依赖口径；fc1/down/lm 本就 DRAM
  封顶不动）。**probe F 三次复跑（唯一裁决）：all-2D(m64u8) 12.03、qkv2D+2D
  11.99 vs all-MPS 12.19 / shipping 12.18 ms/branch——真实但仅 +1.6%，未过
  「明显低于 12.2」门槛**。生产 serve A/B（seed 831001）：gpu busy 27.34→
  25.94s 但墙钟 e2e 29.8→29.9s（采样轨迹分叉 746/747 token，墙钟裁决）——
  **零生产收益**。结论：decode 链已不由 GEMV 带宽决定——probe F 实测 ~12ms
  即 GEMM 内核执行本身（~246GB/s 内核效率地板，M9 口径；"520GB/s 流式地板"
  是物理上限不是内核可达值），派发税另计仅 ~0.6ms/branch-step（M12 斜率桩
  实测）。默认路径不变（qkv 融合 per-column + MPS），2D 内核保留为 env 可选
  变体；semantic 剩余空间在内核执行效率（见已知限制：调度侧已封盘）。

## 数值口径

- **melody_abc 贪心与 lyra 逐位一致**（tests/test_sampling e2e 对拍门）。
- semantic（cot=off legacy bf16 路径）存在 bf16 本征 argmax 翻转带：与 lyra 的
  分歧步 margin ≤ 1.0 即接受（split-K SDPA 下 off_semantic 首翻 step40
  margin=0.0625、melody_semantic step8 margin=0.0625；flash2 回退路径为
  step24 margin=0——翻转步随舍入路径漂移属预期，门不变）。
- 初始噪声为引擎自有 splitmix64 流（M0 决策）：**同 seed 不逐位复刻 lyra 的
  PCG64 抽样**，歌长/旋律分布等价，听感对比靠耳朵。
- bf16 权重、fp32 累加、运算边界 RNE；各阶段对拍门限按「双实现本征噪声带」标定
  （fixture 门限见 tests/，接线错误实测 ≥0.3 量级，区分度 7 倍+）。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `H3_QKV_GEMV` | 开 | =0 禁用 q/k/v 融合 GEMV（回退三连 MPS linear） |
| `H3_GEMV` | yue 默认开 | 单矩阵 rows==1 GEMV 快路；yue 进程默认 `1`+`qw_gemv2d_m64u8`（M14，显式 env 覆盖，`H3_GEMV=0` 回退全 MPS）；qwen 沿用 h3 自身默认（关） |
| `H3_GEMV_KERNEL` | yue 默认 2D | GEMV 变体（u2/u4/u8/c2u4/g128…、2D 块家族 `qw_gemv2d_m4…m64u8`；M14 起 yue 默认 `qw_gemv2d_m64u8`） |
| `H3_GEMV_MAX_OUT` / `H3_GEMV_MIN_IN` | 无 | per-shape 分解旋钮：output_dim ≥ / input_dim < 阈值时走 MPS |
| `H3_QKV_GEMV_KERNEL` | qw_gemv_qkv_bf16_u8 | 融合 GEMV 变体（u4/u8 或 2D `qw_gemv2d_qkv_m32u8`/`m64u8`，后者要求 out_q/out_kv 可整除 row_tile） |
| `H3_ZERO_COPY_WEIGHTS` | 1 | 权重 mmap 零拷贝（不计 footprint、可按需重读）；=0 私有拷贝。M8 实测对 GEMV/MPS 性能均无影响 |
| `YUE_VAE_CPU` | 关 | =1 VAE 解码走 Accelerate CPU 参考路径（GPU 路径对照/救火） |
| `H3_FFMPEG` / `H3_FFPROBE` | 无 | 覆盖 ffmpeg/ffprobe 可执行路径（WAV24→FLAC 转码用，默认取 PATH） |
| `YUE_SDPA_FLASH2` | 关 | =1 decode 步注意力回退 flash2 tile 内核（默认 split-K 标量对：probe U kv=513 134→28.5µs，见性能节 M10） |
| `YUE_HN_ROPE_SEPARATE` | 关 | =1 q/k head-norm+rope 回退四内核路径（默认融合单 dispatch：位级一致，probe T +1.39 ms/branch，见性能节 M11） |
| `YUE_AR_CONTINUE_EVERY` | 1 | 层尾 commit 粒度（每 N 层一次 continue；实测 GPU busy 持平） |
| `YUE_SERVE_MEM` | 关 | =1 每请求边界打印 `[mem]`（live/peak/allocated/device-alloc/footprint）与 `[gpu]` 统计行（A/B 与内存排查标准工具） |
| `YUE_HOST_TRACE` | 关 | =1 每 512 步打印 semantic 每 token 分解 `[host]`（wall/enc/wait/read/guided/unpack/dist/sample，M14 诊断基建） |
| `YUE_CFG_BATCH` | 开 | =0 回退串行双分支 decode（M15 batch fusion 的逃生门） |
| `YUE_BATCH_DBG` | 关 | =1 batched decode 每步扫描 logits 非有限值，命中时打印步号/行分布并 dump /tmp/batch_nan_s*.bin |
| `YUE_NO_PREFILL_PAD` | 关 | =1 关闭 prefill 行数桶化（`yue_ar_forward` 行尾 pad 到 512 行倍，数值逐位不变；内存排查对照用） |
| `YUE_RESTART_AFTER` | 10 | serve 模式每 N 个请求在边界 execve 自重启（同 PID/fd，宿主无感知）归还驱动持有的 Metal 页；=0 关闭 |
| `YUE_ALLOC_TRACE` | 关 | =1 打印 ≥8MB 的 Metal 分配及调用栈（内存归因） |
| `H3_KEEP_GRAPHS` | 关 | =1 跳过请求边界的 MPSGraph 图缓存清除（归因实验用） |
| `H3_DISABLE_MPS_LINEAR` | 关 | 回退标量 16×16 tile 线性内核 |
| `QWEN_AR_TRACE` / `QWEN_AR_TRACE_LAYER0` / `QWEN_AR_DUMP` | 关 | 逐层跟踪 / layer0 dump / 全量 dump（调试） |
| `QWEN_VAE_DUMP` | 关 | VAE 逐阶段激活 dump（fixture 对拍；GPU 路径自动 submit/read） |

## 测试与 fixture

七套回归（全部以 `0 failure(s)` 为过）：`m0`（冒烟）、`tokenizer`（tiktoken 逐位）、
`gemv`（k-映射 + 位级对拍 + qkv 融合）、`ar`（prefill/step parity vs lyra fixture）、
`sampling`（11 单元 case + melody_abc/semantic 贪心 e2e）、`nar`（条件缓存/velocity/
solver/多 chunk）、`vae`（67 阶段 + 音频对拍；默认走 GPU 解码路径，Metal 不可用时
自动回退 CPU 并打印提示，`YUE_VAE_CPU=1` 强制 CPU）。

fixture 生成（需参考栈 venv）：`make tokenizer-fixture ar-fixture
sampling-fixture nar-fixture vae-fixture`（nar-fixture 需
`LYRA_FORCE_BF16_SDPA=1`）。微基准：`make bench-gemv bench-dispatch`；
`./tests/bench_dispatch F` 只跑
probe F（生产节奏 decode 链：真依赖 + 层尾 continue + CFG 双分支 + 2048 步热态，
kernel 优劣的唯一 probe 裁据，结论见文件头注释；`G` = 按形状真依赖带宽校准，
`T`/`U`/`S` = 时间线消融 / SDPA 微基准 / 2D sweep）。

## 架构地图

| 文件 | 职责 |
|---|---|
| `yue_ar.{h,m}` | 28 层 GQA AR backbone：KV cache、RMS/RoPE/SDPA、qkv 融合 GEMV、abc/semantic 相位投影 |
| `yue_sampling.{c,h}` | fp32（abc/CFG）与 legacy-bf16（cot=off）双采样路径、窗口频率幂惩罚、splitmix64 RNG |
| `yue_generate.{c,h}` | AR 生成循环：CFG 双分支（M15 起 batch 单流 `decode_step2`，`YUE_CFG_BATCH=0` 回退串行）、guided_logits |
| `yue_nar.{h,m}` | NAR 声学流匹配：conditioning 预填充复用 AR 缓存、非因果 prefix SDPA、midpoint solver |
| `yue_vae.{c,h}` | Oobleck VAE 解码：GPU 路径（TensorOps matmul2d 逐 tap GEMM + 零 ring，默认）与 Accelerate CPU 参考路径（`YUE_VAE_CPU=1`）、halo 分块、WAV24 + ffmpeg FLAC |
| `yue_gpu_ext.h` | yue/qwen 侧 GPU wrapper 声明（qwen VAE 家族 `qw_*` + yue 声学 VAE 家族 `yv_*`） |
| `yue_protocol.c` / `yue_json.c` | 协议常量/指令模板/token 前缀/JSON 解析 |
| `main.c` | serve（JSONL）+ REPL + GPU 锁 + 引擎生命周期 |
| `h3_*`（vendored） | Metal 运行时/权重/safetensors/tokenizer（MIT，源自 h3-metal，经 qwen-metal 沿用；含 qwen 追加的 `qw_*` 内核与 yue 追加的 `yv_*` 声学 VAE 内核，均在 h3_shaders.metal + h3_gpu.m + pipeline 名单注册） |

里程碑：M0 骨架 → M1 tokenizer → M2 AR parity → M3 采样+生成循环 → M4 NAR →
M5 VAE+FLAC → M6 serve/REPL 全管线 → M7 性能轮（四轮）→ M8 VAE GPU 化 + GEMV
谜团收口 → M9 MLX 式 2D 块 GEMV（二次否决）→ M10 decode split-K SDPA 落地 →
M11 attention prologue 融合 → M12 图捕获裁决（调度封盘）→ M13 GEMM 赛道终审
（满风扇+分档校准，内核封盘）→ M14 宿主尾归因修正 + decode GEMV 默认切 2D
块内核 → M15 CFG batch fusion（semantic −40%，
反超 lyra）。M16/M17 曾实现 `--precision 8bit`（lyra ar-8bit 权重 + int8
GEMV），实测仅省内存不提速（略慢 ~4%），已整体移除。

## 已知限制

- **semantic 分解账（M14 打点口径，batch fusion 前的 ~19ms/token）**：GEMM
  ~10.5-12（流式地板，M13）+ MPSGraph linear encode ~3.1（固有税，三条
  替代路径已全部止损）+ 自研 dispatch 编码 ~1.3 + split SDPA ~2-3（随 KV
  增长，常数项 ~0.5-1ms 是最后可动项）+ 小内核 ~1（已融合到底）+ 宿主
  0.11。斜率桩实测真实链每 dispatch 边际成本 0.89-1.08µs——不存在隐藏
  气泡池；整步 MPSGraph 图捕获结构性不可行（自研内核进不了 MPSGraph 且
  bf16 舍入序被 melody_abc 逐位门锁死）。**decode 链全赛道封盘**
  （调度 M12、内核 M13、编码/宿主 M14）。M15 的 batch fusion 把其中双分支
  重复的 GEMM 权重流并成一遍，semantic 11.7ms/token **反超** lyra 同口径
  ≈15.1（−22%），仅剩 SDPA 常数项可选；flow/save 已持平或反超。
- 语义生成的 per-step top-2 margin 记录、logits 回读与 host 采样造成每 token 一次
  GPU 排空 + ~1.2ms 宿主尾。
- VAE GPU 权重常驻（重排后 ~150MB fp32）；VAE 解码在 64 对齐 tile 上运行，
  输出与 CPU 参考路径在同本征带内（非逐位一致）。
- **连续生成的内存口径（2026-11-02 实测 + 两轮修复）**：h3 张量 live 逐字节
  平台、`MallocStackLogging` 下 `leaks` = 0——无泄漏；footprint 棘轮全部来自
  「形状漂移 × 驱动池」（Metal freed 页不还 OS，驱动池按精确大小复用）。
  已落地五层缓解：①KV cache / NAR chunk 容量按 512 行桶化
  （`YUE_SHAPE_BUCKET`）；②VAE halo tile 的最后一块起点回退（每 tile 输入
  收敛到 {core+halo, core+2*halo}，重叠区由 halo 保证两次解码同值，逐位
  不变）；**③prefill 行数桶化**（`yue_ar_forward` 对 count>1 的前向 pad 到
  512 行倍、行尾填 token 0——因果 limit 下真实行逐位不变，decode count==1
  不 pad；`YUE_NO_PREFILL_PAD=1` 回退；覆盖 semantic/abc/NAR-cond 三个随歌
  变长的形状源）；**④VAE GPU 解码 scratch 跨 tile 提升**（bufs[3]+ring 每歌
  只分配一次——此前每 tile 重建，且 ring 尺寸族彼此只差 KB 级（t×cin 随层
  变化），把 free 出的大块切碎成不可复用的小块，实测每 tile 净增 ~135MB
  级池区域、~2GB/首，是 cot=full 长歌棘轮的主因；提升后 6 首 footprint
  9.7→11.2GB 走平 vs 修复前 12→24.4GB）；⑤**serve 模式
  `YUE_RESTART_AFTER`（默认 10 首）在请求边界 execve 自重启**——同一
  PID、同一 stdin/stdout fd，宿主完全无感知，权重 mmap 惰性重载 ~1s，
  footprint 周期性归零。当前实测（6 首 cot=full 混合歌长连发）：live 恒
  9.08GB、peak 11.8GB、footprint 9.7→11.2GB 走平（残余 ~0.25GB/首为 NAR
  velocity 行数随歌变化的 MPS 形状漂移，自重启兜底）。`allocated` 只增不减
  是累计吞吐口径；泄漏判据看 `YUE_SERVE_MEM=1` 的 `[mem] live` 行（另有
  `device-alloc` = currentAllocatedSize，用于分离「张量簿记」与「驱动仍持有
  的资源」）。对照：MLX 栈（lyra）水位天然恒定——同 seed/不同 seed、不同歌长
  连发 footprint 恒 ~7.55GB（`mx.clear_cache()` 用户态池显式归还 + 逐 kernel
  确定性执行、形状空间稳定；native 的 MPS rows=1 GEMM 有 1-ulp 运行间非确定性，
  混沌解码放大成形状漂移，故需上述五层缓解 + 自重启兜底）。
- 仅 Apple Silicon；权重目录布局需符合 lyra 转换目录铁律（无 symlink、manifest
  零容忍多余文件、ar-bf16/nar-bf16 必需）。
- 非 git 仓库；vendored `linenoise.c` 存在 30 条历史遗留编译警告（自研代码零警告）。
