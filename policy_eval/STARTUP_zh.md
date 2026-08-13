# VLA × LIBERO 评测启动流程

本文档适用于当前目录结构：

```text
D:\BenchmarkTest\
└── LIBERO-PRO-trial\
    ├── policy_eval\
    └── libero\libero\
        ├── bddl_files\    # BDDL 数据
        └── init_files\    # 初始状态数据
```

LIBERO/MuJoCo 环境端运行在 WSL 中；VLA 模型可以运行在同一个 WSL、Windows
主机或远程 GPU 服务器上。实时场景通过浏览器查看。

## 1. 首次创建 `libero-pro-cpu` 环境

本节是环境配置的核心，只需执行一次。它假设 WSL Ubuntu 和 Miniforge 已经安装，
且 Miniforge 位于 `~/miniforge3`。已经能正常激活 `libero-pro-cpu` 的机器可以直接
跳到“每次进入 WSL 后的准备步骤”。

`libero-pro-cpu` 只运行 LIBERO、MuJoCo 和评测器，不加载真实 VLA 模型。因此它
使用 CPU PyTorch，不安装 CUDA；真实 VLA 模型使用独立的模型服务器环境。

### 1.1 安装 MuJoCo 无头渲染系统库

在 WSL 中执行：

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  libosmesa6-dev \
  libgl1-mesa-dev \
  libglfw3
```

OSMesa、Mesa 和 GLFW 用于 WSL 中的 MuJoCo 离屏渲染。如果这些库已经安装，重复
执行不会影响现有环境。

### 1.2 创建并激活 Conda 环境

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda create -n libero-pro-cpu python=3.8 -y
conda activate libero-pro-cpu

python --version
which python
```

当前已验证环境的输出是：

```text
Python 3.8.13
/home/<WSL用户名>/miniforge3/envs/libero-pro-cpu/bin/python
```

如果 `conda create` 提示该环境已经存在，不要覆盖或删除它，直接执行：

```bash
conda activate libero-pro-cpu
```

### 1.3 安装 CPU PyTorch

```bash
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  torch==1.11.0+cpu
```

安装后确认 PyTorch 不使用 CUDA：

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

预期结果包含：

```text
1.11.0+cpu
CUDA available: False
```

### 1.4 安装 LIBERO 和评测依赖

```bash
cd /mnt/d/BenchmarkTest/LIBERO-PRO-trial

python -m pip install -r policy_eval/requirements-env-cpu.txt
python -m pip install --no-deps -e .
```

`requirements-env-cpu.txt` 会安装当前已验证的环境端依赖，包括：

- NumPy 1.24.4
- robosuite 1.4.0
- MuJoCo 3.2.3
- BDDL 1.0.1
- Gym、imageio 和录像支持

`pip install --no-deps -e .` 会把当前仓库以可编辑模式安装为 `libero` 包；之后修改
仓库代码无需重新安装。评测数据也已经包含在仓库内：

```text
libero/libero/bddl_files
libero/libero/init_files
```

正常运行不依赖额外的 `libero_data/`，也不需要传 `--data-root`。

安装后必须检查 `libero` 实际指向当前 `LIBERO-PRO-trial`，避免环境仍引用旧目录：

```bash
python -m pip show libero | grep -E '^(Version|Editable project location):'
```

预期包含：

```text
Version: 0.1.0
Editable project location: /mnt/d/BenchmarkTest/LIBERO-PRO-trial
```

如果这里显示 `LIBERO-PRO-master` 或其他旧目录，在当前 Conda 环境中重新挂载：

```bash
python -m pip uninstall -y libero
cd /mnt/d/BenchmarkTest/LIBERO-PRO-trial
python -m pip install --no-deps -e .
```

### 1.5 检查环境是否安装正确

确认当前 Python 确实来自 `libero-pro-cpu`：

```bash
conda info --envs
which python
python --version
python -m pip show libero | grep -E '^(Version|Editable project location):'
```

检查关键依赖版本和导入：

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
python -c "import torch, numpy, mujoco, robosuite, libero; print('imports: OK'); print('torch:', torch.__version__); print('numpy:', numpy.__version__); print('mujoco:', mujoco.__version__); print('robosuite:', robosuite.__version__)"
```

运行协议和实时预览单元测试：

```bash
python -m unittest tests.test_policy_eval_protocol -v
```

### 1.6 运行两步完整冒烟测试

```bash
bash policy_eval/run_mock_eval.sh \
  --max-steps 2 \
  --stabilization-steps 1 \
  --camera-size 128 \
  --no-save-video \
  --live-preview
```

如果终端打印任务信息、实时预览地址并最终显示 `evaluation completed`，说明
`libero-pro-cpu`、MuJoCo、LIBERO、mock 策略和实时预览链路均已配置成功。mock
不会完成抓取任务，因此 `success_rate=0.0` 是预期结果。

### 1.7 一次性复制版本

在 WSL 和 Miniforge 已安装的前提下，以下命令可以从头创建该环境：

```bash
sudo apt update
sudo apt install -y build-essential libosmesa6-dev libgl1-mesa-dev libglfw3

source ~/miniforge3/etc/profile.d/conda.sh
conda create -n libero-pro-cpu python=3.8 -y
conda activate libero-pro-cpu

python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  torch==1.11.0+cpu

cd /mnt/d/BenchmarkTest/LIBERO-PRO-trial
python -m pip install -r policy_eval/requirements-env-cpu.txt
python -m pip install --no-deps -e .
python -m pip show libero | grep -E '^(Version|Editable project location):'

python -m unittest tests.test_policy_eval_protocol -v
```

## 2. 每次进入 WSL 后的准备步骤

先在 Windows PowerShell 中进入 WSL：

```powershell
wsl
```

然后在 WSL 中激活已经安装好的 Conda 环境并进入项目：

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate libero-pro-cpu
cd /mnt/d/BenchmarkTest/LIBERO-PRO-trial

python --version
```

预期显示：

```text
Python 3.8.13
```

再确认仓库内数据目录存在：

```bash
test -d libero/libero/init_files && echo "init data: OK"
test -d libero/libero/bddl_files && echo "bddl data: OK"
```

评测器默认从这两个仓库内目录加载数据，不需要额外下载或传入 `--data-root`。

如果出现 `Command 'python' not found`，说明当前终端没有激活 Conda 环境。重新执行：

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate libero-pro-cpu
```

不要直接改用系统 `python3`，因为系统环境通常没有安装 LIBERO、MuJoCo 和
robosuite 所需的版本。

## 3. 推荐：一条命令启动 mock 联调

首次检查环境、动作协议和实时画面时，推荐使用 mock。该脚本会自动启动 mock
模型服务、运行评测，并在退出时关闭模型服务：

```bash
bash policy_eval/run_mock_eval.sh \
  --max-steps 50 \
  --stabilization-steps 2 \
  --camera-size 128 \
  --no-save-video \
  --live-preview
```

看到类似输出后：

```text
live preview: http://127.0.0.1:8765/
```

在 Windows 浏览器打开该地址，即可实时查看：

- `agentview` 主相机；
- 机械臂腕部相机；
- episode、动作步、策略请求次数；
- 成功状态和推理延迟。

mock 的 `noop` 策略不会抓取物体，因此最终 `success_rate=0.0` 是预期结果。
它只用于确认整条评测链路可以运行。

如果当前终端无法使用 `python`，也可以临时明确指定 Conda Python：

```bash
PYTHON_BIN="$HOME/miniforge3/envs/libero-pro-cpu/bin/python" \
bash policy_eval/run_mock_eval.sh \
  --max-steps 50 \
  --no-save-video \
  --live-preview
```

## 4. 双终端启动方式

真实 VLA 评测由两个独立进程组成：

```text
终端 A：VLA 模型服务器
终端 B：LIBERO/MuJoCo 环境评测器
```

### 4.1 终端 A：启动模型服务

验证流程时可以先启动仓库自带的 mock 服务：

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate libero-pro-cpu
cd /mnt/d/BenchmarkTest/LIBERO-PRO-trial

python -m policy_eval.mock_policy_server \
  --host 127.0.0.1 \
  --port 8000 \
  --mode noop \
  --action-horizon 5
```

服务就绪时会输出：

```text
mock policy ready: http://127.0.0.1:8000 model=mock/noop horizon=5
```

可在另一个 WSL 终端检查：

```bash
curl http://127.0.0.1:8000/healthz
```

真实 pi0、pi0.5、SmolVLA 或其他模型应启动各自的服务适配器，并实现：

- `GET /healthz`
- `GET /v1/metadata`
- `POST /v1/actions`

模型服务必须返回可直接交给 LIBERO `env.step()` 的 `[T, 7]` 动作块。

### 4.2 终端 B：启动 LIBERO 评测

重新打开一个 WSL 终端，并再次激活环境：

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate libero-pro-cpu
cd /mnt/d/BenchmarkTest/LIBERO-PRO-trial
```

运行评测：

```bash
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
  --live-preview \
  --save-video
```

评测结束后结果默认写入：

```text
outputs/policy_eval/<suite>_<task>_<timestamp>/
├── results.json
└── episode_000_init_000.mp4
```

## 5. 实时预览和录像选项

实时预览和录像互相独立：

```bash
# 实时查看并录制最终视频
--live-preview --save-video

# 只实时查看，不录制视频
--live-preview --no-save-video

# 不实时查看，只录制最终视频
--save-video
```

实时预览可调参数：

```bash
--live-preview-port 8765 \
--live-preview-fps 10 \
--live-preview-stride 1
```

- `--live-preview-fps`：浏览器每秒拉取画面的次数。
- `--live-preview-stride N`：每 N 个环境动作步编码一次画面；增大它可降低 CPU
  开销。
- `--live-preview-port`：预览页面端口；若 8765 被占用，可改为其他端口。
- 评测结束后预览服务会自动关闭，最终留档仍以 MP4 和 `results.json` 为准。

默认仅监听 `127.0.0.1`。如果必须让另一台机器访问，可增加：

```bash
--live-preview-host 0.0.0.0
```

此预览服务没有身份验证或 TLS，只应在可信内网或 SSH 隧道中使用，不应直接暴露
到公网。

## 6. 连接真实或远程 VLA 服务器

如果 VLA 服务运行在远程 GPU 服务器，将 `--policy-url` 改为该服务器的内网地址：

```bash
python -m policy_eval.eval_one_task \
  --policy-url http://192.168.1.50:8000 \
  --suite libero_object \
  --task-name pick_up_the_cream_cheese_and_place_it_in_the_basket \
  --live-preview
```

先在 WSL 中检查连通性：

```bash
curl http://192.168.1.50:8000/healthz
```

若模型服务运行在 Windows 主机而 WSL 使用 NAT 网络，WSL 中的 `127.0.0.1`
不一定代表 Windows 主机。可先查询 Windows 主机在 WSL 侧的地址：

```bash
WINDOWS_HOST_IP=$(ip route show default | awk '{print $3}')
echo "$WINDOWS_HOST_IP"
curl "http://${WINDOWS_HOST_IP}:8000/healthz"
```

然后使用：

```bash
--policy-url "http://${WINDOWS_HOST_IP}:8000"
```

Windows 模型服务还需要监听可被 WSL 访问的网卡，并允许对应防火墙端口。不要将
未鉴权的模型服务直接暴露到公网。

## 7. 常见问题

### `python: command not found`

当前 WSL 终端未激活 Conda：

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate libero-pro-cpu
```

### `conda: command not found`

先加载 Miniforge 的 shell 脚本：

```bash
source ~/miniforge3/etc/profile.d/conda.sh
```

然后再执行 `conda activate libero-pro-cpu`。

### WSL 提示 localhost 代理未镜像

例如：

```text
wsl: 检测到 localhost 代理配置，但未镜像到 WSL。
NAT 模式下的 WSL 不支持 localhost 代理。
```

这条消息表示 Windows 上配置的 localhost HTTP 代理没有自动传入 WSL。对于
同一 WSL 内运行的 mock 服务和 LIBERO 评测没有影响，可以忽略。它可能影响
`pip`、Git 或访问远程模型服务；如遇网络问题，再单独配置一个 WSL 可访问的代理
地址。

### 无法连接 `127.0.0.1:8000`

确认模型服务所在终端仍在运行：

```bash
curl http://127.0.0.1:8000/healthz
```

若失败，先启动模型服务，或者检查 `--policy-url` 是否指向了正确机器。

### 8000 或 8765 端口被占用

查看监听端口：

```bash
ss -ltnp | grep -E ':(8000|8765)\b'
```

模型服务和评测器两端需要使用一致的模型端口；预览端口则可单独修改：

```bash
--live-preview-port 8766
```

### 如何停止评测

在运行评测的 WSL 终端按 `Ctrl+C`。一键 mock 脚本会同时清理它启动的 mock
服务；双终端方式还需要在模型服务终端按一次 `Ctrl+C`。

## 8. 最短启动清单

只想快速看到实时画面时，依次执行：

```powershell
wsl
```

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate libero-pro-cpu
cd /mnt/d/BenchmarkTest/LIBERO-PRO-trial

bash policy_eval/run_mock_eval.sh \
  --max-steps 50 \
  --camera-size 128 \
  --no-save-video \
  --live-preview
```

最后在 Windows 浏览器打开终端打印的 `live preview` 地址。
