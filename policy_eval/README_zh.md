# LIBERO 环境端与模型端分离评测

完整的逐步启动命令与故障排查见
[`STARTUP_zh.md`](STARTUP_zh.md)。

这套代码把评测拆成两个互相独立的进程：

```text
WSL / CPU 环境端                                模型服务器 / GPU 端
┌──────────────────────────────┐                ┌─────────────────────────┐
│ LIBERO + MuJoCo              │  HTTP JSON     │ pi0 / pi0.5 / SmolVLA  │
│ reset → 取观测 → env.step    │ ─────────────> │ 图像/状态预处理 + 推理   │
│ 成功判定、280 步上限、录像   │ <───────────── │ 反归一化后返回 [T, 7]    │
└──────────────────────────────┘                └─────────────────────────┘
```

本地 WSL 不加载模型，不需要 CUDA，也不用为 LIBERO-PRO 配置旧 CUDA。模型端以后只需实现相同 HTTP 接口，环境端代码不变。

## 已验证的任务

- Suite：`libero_object`
- Task：`pick_up_the_cream_cheese_and_place_it_in_the_basket`
- 初始状态：`0`
- 正式预算：280 个动作步；动作块长度 5；每 5 步重新请求
- Mock 策略：`mock/noop`，返回 `[0, 0, 0, 0, 0, 0, -1]`

Mock 不会抓起物体，因此 `success_rate=0.0` 是预期结果。它验证的是完整评测链路能否跑通，而不是策略能力。

## 1. WSL 环境

在 WSL 中进入 Windows 项目目录：

```bash
cd /mnt/d/BenchmarkTest/LIBERO-PRO-trial
source ~/miniforge3/etc/profile.d/conda.sh
conda activate libero-pro-cpu
```

如果还没有这个环境，可用 Python 3.8 创建。PyTorch 只装 CPU 版：

```bash
conda create -n libero-pro-cpu python=3.8 -y
conda activate libero-pro-cpu
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  torch==1.11.0+cpu
```

安装无头渲染系统库：

```bash
sudo apt update
sudo apt install -y libosmesa6-dev libgl1-mesa-dev libglfw3
```

安装最小环境侧 Python 依赖，并挂载当前源码：

```bash
python -m pip install -r policy_eval/requirements-env-cpu.txt
python -m pip install --no-deps -e .
```

若官方 PyPI 下载很慢，可以只给这次安装使用清华镜像：

```bash
python -m pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -r policy_eval/requirements-env-cpu.txt
```

评测数据已经包含在仓库中，不依赖仓库外的 `libero_data/`：

- BDDL：`libero/libero/bddl_files`
- 初始状态：`libero/libero/init_files`

正常运行不需要传 `--data-root`。只有在部署时主动把数据放到其他目录，才使用
`--data-root /path/to/data` 覆盖上述默认路径。

它会在 `.runtime/libero_config/config.yaml` 自动写入正确的 Linux 路径，不触发 LIBERO 首次导入时的交互式提问。

## 2. 启动 mock 模型服务（终端 A）

```bash
cd /mnt/d/BenchmarkTest/LIBERO-PRO-trial
source ~/miniforge3/etc/profile.d/conda.sh
conda activate libero-pro-cpu

python -m policy_eval.mock_policy_server \
  --host 127.0.0.1 \
  --port 8000 \
  --mode noop \
  --action-horizon 5
```

看到下面这行就说明服务已启动：

```text
mock policy ready: http://127.0.0.1:8000 model=mock/noop horizon=5
```

可在另一个终端检查：

```bash
curl http://127.0.0.1:8000/healthz
```

## 3. 跑一次完整 task eval（终端 B）

```bash
cd /mnt/d/BenchmarkTest/LIBERO-PRO-trial
source ~/miniforge3/etc/profile.d/conda.sh
conda activate libero-pro-cpu

python -m policy_eval.eval_one_task \
  --policy-url http://127.0.0.1:8000 \
  --suite libero_object \
  --task-name pick_up_the_cream_cheese_and_place_it_in_the_basket \
  --n-episodes 1 \
  --init-state-ids 0 \
  --max-steps 280 \
  --stabilization-steps 10 \
  --action-horizon 5 \
  --replan-steps 5 \
  --camera-size 256 \
  --save-video
```

### 实时查看场景

LIBERO 的 `OffScreenRenderEnv` 会在每个 `env.step()` 后返回最新相机观测。加上
`--live-preview` 后，评测器会把这些观测发布到本地浏览器页面，不需要等待 MP4
写完，也不要求桌面 OpenGL 窗口：

```bash
python -m policy_eval.eval_one_task \
  --policy-url http://127.0.0.1:8000 \
  --suite libero_object \
  --task-name pick_up_the_cream_cheese_and_place_it_in_the_basket \
  --live-preview
```

启动后终端会打印类似地址：

```text
live preview: http://127.0.0.1:8765/
```

在 Windows 浏览器打开该地址，可以同时看到 `agentview`、腕部相机、episode、
步数、策略请求数、成功状态和 HTTP 推理延迟。页面默认每秒刷新 10 次，环境端
默认每个动作步发布一次：

```bash
python -m policy_eval.eval_one_task \
  --live-preview \
  --live-preview-port 8765 \
  --live-preview-fps 10 \
  --live-preview-stride 1
```

- `--live-preview-fps` 控制浏览器拉取画面的频率。
- `--live-preview-stride N` 每 N 个环境动作步编码一次画面，可降低 CPU 开销。
- 实时预览与录像相互独立；可同时使用，也可加 `--no-save-video` 只看实时画面。
- 默认只监听 `127.0.0.1`。跨机器查看时可设 `--live-preview-host 0.0.0.0`，但应
  只在受信内网或经 SSH 隧道访问；预览服务本身不提供身份验证或 TLS。
- 评测结束后服务随进程关闭，最后一帧不会永久托管；最终留档仍使用 MP4 和
  `results.json`。

不写 `--run-name` 时，程序自动用时间生成新目录，因此不会覆盖旧结果。先做快速检查时可改为：

```bash
python -m policy_eval.eval_one_task \
  --policy-url http://127.0.0.1:8000 \
  --max-steps 5 \
  --camera-size 128 \
  --no-save-video
```

也可以在已经激活 Conda 环境后，用一条命令临时启动 mock、跑评测并自动关闭 mock：

```bash
bash policy_eval/run_mock_eval.sh \
  --max-steps 5 \
  --stabilization-steps 2 \
  --camera-size 128 \
  --no-save-video
```

这个一键脚本只适合 mock 联调；接远程真实模型时仍使用前面的评测器命令。

## 4. 输入输出协议

环境端调用以下接口：

- `GET /healthz`：服务是否就绪
- `GET /v1/metadata`：模型名、动作维度和约定
- `POST /v1/actions`：用当前观测请求一个动作块

每次动作请求包含：

- 任务名、英文指令、episode、初始状态 ID 和当前步数
- `agentview_rgb` 与 `wrist_rgb`：朝向已校正的 RGB `uint8` 图像
- 末端位置、四元数、夹爪关节位置
- 常用的 8 维状态：`xyz + axis-angle + 两个夹爪关节`
- 希望返回的动作块长度

服务端必须返回形状为 `[T, 7]` 的有限浮点数：

```text
[dx, dy, dz, dRx, dRy, dRz, gripper]
```

每个值必须在 `[-1, 1]`。返回值必须已经完成该模型所需的反归一化，可以直接传给 LIBERO 的 `env.step()`。模型特有的图像裁剪、state 组织、prompt 格式、归一化和反归一化都放在服务器适配器中。

当前 JSON + base64 协议优先保证清楚和可调试；接入真实 GPU 服务器后，如果网络吞吐成为瓶颈，可保持字段语义不变，再换 JPEG/二进制传输。

## 5. 输出

默认写到：

```text
outputs/policy_eval/<suite>_<task>_<timestamp>/
├── results.json
└── episode_000_init_000.mp4
```

`results.json` 包含环境配置、实际 BDDL/init 路径、包版本、策略服务器元数据、每个 episode 的成功与否、步数、请求次数、每次 HTTP/推理延迟及聚合成功率。

## 6. 换成真实模型服务器

真实服务器实现同样三个接口即可。服务器监听远程网卡时使用：

```bash
python your_model_server.py --host 0.0.0.0 --port 8000
```

WSL 环境端只改地址：

```bash
python -m policy_eval.eval_one_task \
  --policy-url http://服务器IP:8000 \
  --suite libero_object \
  --task-name pick_up_the_cream_cheese_and_place_it_in_the_basket
```

`0.0.0.0:8000` 不应直接暴露到公网；跨机器测试应使用防火墙、内网或带鉴权/TLS 的反向代理。

## 7. 协议测试

协议测试不依赖 LIBERO 或 MuJoCo，在 Windows/WSL 都可运行：

```bash
python -m unittest tests.test_policy_eval_protocol -v
```
