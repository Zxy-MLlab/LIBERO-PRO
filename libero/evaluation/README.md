# LIBERO Policy Evaluation

环境端与远程 VLA 推理端的完整启动、冒烟测试和排错流程见
[`STARTUP_zh.md`](STARTUP_zh.md)。

## Quick Start

安装额外需要的依赖：

```bash
conda activate libero_pro
pip install -r libero/evaluation/requirements.txt
```

运行测评：

```bash
python -m libero.evaluation.eval policy=mock benchmark.task_ids='[0]' benchmark.episodes_per_task=1
```

## 常用配置

```bash
python -m libero.evaluation.eval policy=pi0 \
  policy.connection.host=127.0.0.1 policy.connection.port=8000 \
  rollout.execute_horizon=8 \
  benchmark.evaluation_config_path=evaluation_config.yaml benchmark.task_ids='[0,1]' \
  benchmark.init_state_ids='[0,2]' \
  recording.enabled=true live_preview.enabled=true \
  output.directory=outputs/my_eval
```

### GR00T TCP Server

针对以下 GR00T 服务，使用内置的 ZeroMQ TCP client：

```bash
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path $HF_CKPT/GR00T-N1.7-LIBERO/libero_10 \
  --embodiment-tag LIBERO_PANDA --device cuda:0 \
  --host 127.0.0.1 --port 8001 --use-sim-policy-wrapper
```

```bash
python -m libero.evaluation.eval policy=gr00t \
  policy.connection.host=127.0.0.1 policy.connection.port=8001 \
  benchmark.task_ids='[0,1]' benchmark.episodes_per_task=50 \
  output.directory=outputs/gr00t_eval
```

配置文件是 `configs/policy/gr00t.yaml`。它将 evaluator 的 LIBERO observation 转为
GR00T sim wrapper 的 `video.*`、`state.*` 和语言字段，并将 `action.x` 到
`action.gripper` 合成为 `[T, 7]` 的 `delta_ee` 动作块。夹爪输出会复现 GR00T
原生 LIBERO wrapper 的变换：模型的 `0=close / 1=open` 转为 LIBERO 的
`+1=close / -1=open`。

服务以 `--host 127.0.0.1` 启动时只能从同一机器连接。若 evaluator 在另一台机器，
请在 evaluator 机器创建隧道后仍配置 `127.0.0.1:8001`：

```bash
ssh -N -L 8001:127.0.0.1:8001 <user>@<gr00t-server-host>
```

默认使用 MuJoCo EGL 离屏渲染。只有系统安装了 OSMesa 时才覆盖 `environment.render_backend=osmesa`。

`benchmark.init_state_ids=[]` 时，每个 task 根据 `episodes_per_task` 顺序选择不重复初态；默认配置只运行 1 个初态，方便 smoke test。完整评测可设置 `benchmark.episodes_per_task=50`。请求数量超过可用初态时会报错，不会循环复用。显式提供列表时，该列表直接决定 episode schedule 和数量。每个 task 环境固定使用 `benchmark.seed`（默认 7），不会在 episode 之间重新派生 seed。

## Perturbation

扰动由外部 `evaluation_config.yaml` 统一配置，Evaluator 只接收它的路径：

```bash
python -m libero.evaluation.eval \
  benchmark.evaluation_config_path=evaluation_config.yaml
```

配置中用 `task_suite_name` 指定基础 suite，并通过 `use_*` 开启扰动：

```yaml
task_suite_name: libero_goal

use_environment: true
use_swap: false
use_object: false
use_language: false
use_task: false

perturbation_mapping:
  use_environment: env
  use_swap: swap
  use_object: object
  use_language: lan
  use_task: task
```

单扰动的最终 suite 名称由 `task_suite_name` 和 `perturbation_mapping` 组成，例如上面的配置对应 `libero_goal_env`。多个非 task 扰动组合时使用 `libero_goal_temp`；`use_task` 不能与其他扰动同时启用。

启动时，[`perturbation.py`](perturbation.py) 会读取这份配置：

- BDDL 和 `.pruned_init` 都存在时直接复用；
- 只有 `.pruned_init` 缺失时，仅运行 `generate_init_states.py`；
- BDDL 缺失时，先运行对应 Perturbator，再生成 `.pruned_init`。

生成完成后，任务、语言、目标条件和初始状态均由 LIBERO 原生 benchmark 和 BDDL parser 加载。根目录的 `perturbation.py` 仅保留为原仓库示例，evaluation 不会调用它。

## Helper

### Mock Policy Server

可以使用 mock 的 policy 来离线测试评测联通性，默认 `noop` mock 保持机械臂不动，夹爪重复开合。

```bash
python -m libero.evaluation.mock_server --mode noop --chunk-size 16
```

### Record & Live-Preview & Video Render

通过开启 `configs/eval.yaml` 中 `recording` 和 `live_preview` 可以打开视频保存和浏览器在线预览功能。

为提高录制视频的可视化效果，额外提供一个脚本加工录制的视频，Usage: 

```bash
python scripts/render_eval_videos.py \
    outputs/2026-08-14/11-16-46/evaluation
```

默认使用 MuJoCo EGL 离屏渲染。只有系统安装了 OSMesa 时才覆盖 `environment.render_backend=osmesa`。

`benchmark.init_state_ids=[]` 时，每个 task 根据 `episodes_per_task` 顺序选择初态；显式提供列表时，该列表直接决定 episode schedule 和数量。

## OpenVLA 官方 REST 服务

OpenVLA 的推荐入口统一为新版 `libero/evaluation`；远端主分支已经删除旧
`policy_eval` 实现。

环境端使用 OpenVLA 官方 `deploy.py` 的 `POST /act`，不要求 GPU 服务增加本项目的
协议外壳。评测程序不会启动、重启或停止远端 VLA 进程；为避免显存或进程生命周期问题，
`deploy.py` 始终由用户在 GPU 服务器上手动管理。先安装客户端依赖：

```bash
pip install -r libero/evaluation/requirements.txt
```

GPU 服务启动后，在 LIBERO 机器运行：

如果通过 SSH 本地端口转发，在 WSL 终端登录远端并保持该连接：

```bash
ssh -o ExitOnForwardFailure=yes -L 8000:localhost:8000 allinai2
```

登录远端后在同一个终端启动绑定 `0.0.0.0:8000` 的 OpenVLA 服务。另开一个本地
WSL 终端先检查隧道：

```bash
curl --fail http://127.0.0.1:8000/openapi.json
```

随后启动 LIBERO；客户端始终连接本地隧道地址 `127.0.0.1:8000`：

```bash
python -m libero.evaluation.eval policy=openvla \
  policy.connection.base_url=http://127.0.0.1:8000 \
  policy.inference.unnorm_key=libero_10 \
  benchmark.evaluation_config_path=evaluation_config.yaml benchmark.task_ids='[0]' \
  benchmark.episodes_per_task=1 rollout.execute_horizon=1
```

本地 `OpenVLAAdapter` 只使用 `agentview_image`，请求包含 `image`、`instruction` 和配置中
选择的 `unnorm_key`；wrist 图像和机器人 state 不发送。

图像预处理只有两个模式：

- 默认 `policy.adapter.image_preprocess=official_libero`：按官方 LIBERO 顺序旋转 180 度，
  JPEG quality=95 往返和 224×224 Lanczos 缩放。这里使用轻量 Pillow，处理含义与官方
  TensorFlow 实现一致，但不承诺逐像素完全相同。
- 可选 `policy.adapter.image_preprocess=server`：只旋转 180 度，保持原分辨率，由远端
  Hugging Face processor 缩放；该模式不等同于完整的官方 LIBERO 图像路径。

OpenVLA 官方 LIBERO 微调 checkpoint 使用随机裁剪增强，因此设置
`policy.adapter.center_crop=true`。

官方 LIBERO 微调 checkpoint 的本地命令示例：

```bash
python -m libero.evaluation.eval policy=openvla \
  policy.inference.unnorm_key=libero_10 \
  policy.adapter.image_preprocess=official_libero \
  policy.adapter.center_crop=true \
  policy.connection.base_url=http://127.0.0.1:8000 \
  benchmark.evaluation_config_path=evaluation_config.yaml benchmark.task_ids='[0]' \
  benchmark.episodes_per_task=1 rollout.execute_horizon=1 \
  rollout.max_steps=520
```

官方服务成功时经 `json-numpy` 直接返回形状 `[7]` 的 NumPy action，不返回
`{"action": ...}`。Adapter 在本地完成一次 OpenVLA 夹爪符号转换，再交给统一
`PolicyResponse` 校验并传入 `env.step()`。基础 OpenVLA 每次只产生一个动作，因此
建议使用 `rollout.execute_horizon=1`。

动作处理只有官方 LIBERO 路径：前六维不缩放、不换轴、不翻转符号；只把夹爪从
`[0,1]` 映射并二值化到 LIBERO 的符号约定，然后交给 `env.step()`。本地不再提供
Bridge 到 OSC 的实验性放大路径。

`unnorm_key` 必须以 `libero_` 开头，并且必须是远端当前 checkpoint 的
`dataset_statistics.json` 或 `norm_stats` 中实际存在的键。当前 LIBERO-10 checkpoint
使用 `libero_10`；切换到其他 LIBERO checkpoint 时，必须同步改成该 checkpoint 的
实际键，例如 `libero_object`。

`OpenVLAClient.check()` 使用 FastAPI 自动生成的 `/openapi.json` 检查 `/act`，GPU
端不需要额外实现 health endpoint。官方服务推理异常时通常以 HTTP 200 返回字符串
`"error"`，客户端会将其作为评测错误并提示检查服务器日志。

## 动作诊断日志

评测默认在 `${output.directory}/action_traces/` 下为每个 episode 写一个 JSONL。每一行
对应一次真正传入 `env.step()` 的动作，包含 OpenVLA 原始输出、本地转换后的动作、动作
前后末端位置与夹爪状态、实际末端位移、图像统计和请求延迟。该日志完全在 LIBERO 本地
生成，不要求修改远端 `/act` 服务；每行也记录 `unnorm_key` 和 `center_crop`，便于
排除 checkpoint 或图像预处理配置混用。

先用 30 步定位“动作过小或来回抖动”，无需每次跑满整个任务：

```bash
python -m libero.evaluation.eval policy=openvla \
  benchmark.evaluation_config_path=evaluation_config.yaml benchmark.task_ids='[0]' \
  benchmark.episodes_per_task=1 rollout.execute_horizon=1 \
  rollout.max_steps=30 recording.enabled=false live_preview.enabled=false \
  output.directory=outputs/openvla_action_diagnosis
```

如不需要逐步日志，可设置 `diagnostics.action_trace.enabled=false`。


## 架构

Evaluator 先准备扰动数据，再通过 LIBERO 创建环境并执行 rollout。`PolicyClient` 负责与策略服务通信及模型输入输出转换。

```text
evaluation_config.yaml
        ↓
perturbation.py（复用或生成 BDDL / .pruned_init）
        ↓
LIBERO benchmark + environment
        ↓ observation                         ↑ action
EvaluationRunner → ActionChunkExecutor → PolicyClient → Policy Server
```

Evaluator 管理环境生命周期、episode schedule、action chunk 执行、录像和结果统计。`PolicyClient` 管理连接、wire protocol、图像与状态预处理及 action decoding；双方只通过 `PolicyRequest` 和 `PolicyResponse` 交换数据。

## 新增 Policy

### A. 复用已有的传输协议

比如使用 OpenPI 系列的 policy，只需复制 `configs/policy/pi0.yaml` 为 `configs/policy/<name>.yaml`，修改连接和推理配置，随后以 `policy=<name>` 运行。

### B. 使用新的传输协议

新增传输协议应新增client负责数据翻译 `clients/<name>_client.py`，用 `@register_client("<name>")` 注册并实现：

```python
class MyClient(PolicyClient):
    @classmethod
    def from_config(cls, cfg): ...
    def check(self) -> ClientInfo: ...
    def reset(self, episode_id, instruction): ...
    def infer(self, request) -> PolicyResponse: ...
    def close(self): ...
```

Libero-pro协议定义在 `protocol.py` ，规范 evaluator 内部数据结构。Client 完全控制 wire protocol、传输、序列化、图像与 state 预处理、归一化和 action decoding。Evaluator 通过 `rollout.execute_horizon` 决定一个 chunk 最多执行多少个动作；OpenPI 返回的 chunk 长度由服务端模型配置决定。

Action 必须是非空、有限、位于 `[-1, 1]` 的 `float32[T, 7]`，类型为 `delta_ee`。非法响应只终止当前 episode；OpenPI 的连接和请求行为由官方 `WebsocketClientPolicy` 管理，episode 总 timeout 来自 `rollout.episode_timeout_seconds`。

## 常见问题

### PyTorch 无法加载 executable stack

如果出现 `libtorch_cpu.so: cannot enable executable stack`，可使用仓库脚本清除错误 ELF 标志；脚本会先创建备份：

```bash
python scripts/clear_elf_execstack.py
```

### OSMesa 无法加载

如果出现 `libOSMesa.so.0: cannot open shared object file`，使用本机已验证的 EGL：

```bash
python -m libero.evaluation.eval policy=mock environment.render_backend=egl
```
