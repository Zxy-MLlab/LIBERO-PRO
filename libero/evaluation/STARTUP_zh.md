# LIBERO 环境端与远程 VLA 推理端全流程启动说明

本文档说明如何在两台机器上完成一次端到端评测：

- 本地 WSL/CPU：运行 LIBERO、MuJoCo、任务初始化、录像、成功判定和环境侧 Adapter；
- 远程 GPU：只运行 VLA checkpoint 和官方推理服务；
- 两端通过 SSH 本地端口转发通信。

当前推荐的环境端入口是：

```bash
python -m libero.evaluation.eval
```

`policy_eval/eval_one_task.py` 是早期 HTTP Adapter 方案，远端主分支已经将整个
`policy_eval` 目录删除。新测试统一使用 `libero/evaluation`。

## 1. 当前 OpenVLA 数据流

```text
LIBERO task + init state
        │
        ▼
OffScreenRenderEnv 生成原始观测
        │
        ▼
本地 OpenVLAAdapter
  - 取 agentview_image
  - 将仿真图像旋转 180 度
  - JPEG-95、224×224 和可选 center crop
        │
        ▼
POST /act
  {image, instruction, unnorm_key}
        │
        ▼
远端官方 deploy.py + OpenVLA checkpoint
  - processor/tokenizer
  - VLA 推理
  - 按 unnorm_key 反归一化
        │
        ▼
远端直接返回 7 维 NumPy action
        │
        ▼
本地 Adapter 只转换一次夹爪符号
  - 前六维不缩放、不换轴、不翻转
        │
        ▼
env.step(action) → check_success() → 日志/视频/统计
```

`unnorm_key` 由本地配置并随请求发送，真正使用统计量完成反归一化的是远端模型。
它必须与远端 checkpoint 中的键完全一致。例如当前 LIBERO-10 微调模型使用
`libero_10`，不能写成 `libero` 或 `bridge_orig`。

## 2. 路径和终端安排

仓库路径：

```text
Windows: D:\BenchmarkTest\LIBERO-PRO-trial
WSL:     /mnt/d/BenchmarkTest/LIBERO-PRO-trial
```

建议使用两个终端：

| 终端 | 所在机器 | 用途 |
|---|---|---|
| A | 从 WSL SSH 登录远端 GPU | 保持 SSH 隧道并手动运行 `deploy.py` |
| B | 本地 WSL | 检查服务并运行 LIBERO evaluator |

远端服务由使用者手动启动和停止。环境端程序不会启动、重启或结束远端进程。

## 3. 本地 WSL 准备

每次打开 WSL 后执行：

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate libero-pro-cpu
cd /mnt/d/BenchmarkTest/LIBERO-PRO-trial
```

首次使用新版 evaluator 时补齐依赖：

```bash
python -m pip install -r libero/evaluation/requirements.txt
python -m pip install --no-deps -e .
```

确认当前 Python 和仓库：

```bash
which python
python --version
python -m pip show libero | grep -E '^(Version|Editable project location):'
test -d libero/libero/bddl_files && echo "BDDL: OK"
test -d libero/libero/init_files && echo "init states: OK"
```

正常情况下 editable location 指向当前 `LIBERO-PRO-trial`，并显示两项 `OK`。

### 选择原始 LIBERO 或 LIBERO-PRO 任务

新版主分支通过根目录 `evaluation_config.yaml` 选择基础任务族和扰动，不再使用旧参数
`benchmark.suite=...`。

关键字段示例：

```yaml
task_suite_name: libero_10

use_environment: false
use_swap: false
use_object: false
use_language: false
use_task: false
```

五个 `use_*` 全为 `false` 时评测原始 `libero_10`。例如将
`use_environment: true` 后会准备并评测环境扰动 suite，最终名称由
`perturbation_mapping` 决定，通常为 `libero_10_env`。

这里改变的是考题，不是模型训练统计量。评测 `libero_10_env` 时，LIBERO-10 微调模型的
`unnorm_key` 仍然是 `libero_10`。

## 4. 终端 A：建立隧道并启动远端 OpenVLA

在本地 WSL 中登录远端，同时把本地 `127.0.0.1:8000` 转发到远端
`127.0.0.1:8000`：

```bash
ssh -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8000:127.0.0.1:8000 \
  allinai2
```

登录远端后，先检查微调模型统计量。这个检查不加载模型权重：

```bash
export OPENVLA_MODEL=/mnt/ssd1/wangqiyuan/hf-checkpoints/openvla-7b-finetuned-libero-10

python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OPENVLA_MODEL"])
config = json.loads((root / "config.json").read_text())
stats = json.loads((root / "dataset_statistics.json").read_text())

print("config norm_stats:", list(config.get("norm_stats", {})))
print("dataset_statistics:", list(stats))
assert "libero_10" in stats
PY
```

两处最好都显示 `libero_10`。至少 `dataset_statistics.json` 必须包含它；否则不要用
`libero_10` 发起评测。若运行时仍打印一长串 Bridge/Open-X 数据集键，说明远端运行中
的 `self.vla.norm_stats` 没有加载微调统计量，而不是本地 Adapter 出错。

手动启动推理服务：

```bash
cd /home/wangqiyuan/openvla
conda activate openvla

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=1 \
python vla-scripts/deploy.py \
  --openvla_path /mnt/ssd1/wangqiyuan/hf-checkpoints/openvla-7b-finetuned-libero-10 \
  --host 0.0.0.0 \
  --port 8000
```

预期看到：

```text
Uvicorn running on http://0.0.0.0:8000
```

`0.0.0.0` 只是远端进程的监听地址，不是本地 LIBERO 要填写的地址。本地始终连接
SSH 隧道的 `http://127.0.0.1:8000`。

`CUDA_VISIBLE_DEVICES=1` 会让物理 GPU 1 在该进程内部显示为 `cuda:0`，属于正常行为。

## 5. 终端 B：检查远端服务

保持终端 A 不关闭，另开一个本地 WSL 终端：

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate libero-pro-cpu
cd /mnt/d/BenchmarkTest/LIBERO-PRO-trial

curl --fail http://127.0.0.1:8000/openapi.json
```

返回内容中应存在 `/act`。这一步只证明 HTTP 路由可达，不证明 checkpoint、
`unnorm_key` 或模型推理正确。

## 6. 先跑一步冒烟测试

第一次连接新 checkpoint 时只执行一个模型动作：

```bash
python -m libero.evaluation.eval \
  policy=openvla \
  policy.connection.base_url=http://127.0.0.1:8000 \
  policy.inference.unnorm_key=libero_10 \
  policy.adapter.image_preprocess=official_libero \
  policy.adapter.center_crop=true \
  benchmark.evaluation_config_path=evaluation_config.yaml \
  benchmark.task_ids='[0]' \
  benchmark.episodes_per_task=1 \
  rollout.execute_horizon=1 \
  rollout.max_steps=1 \
  recording.enabled=false \
  live_preview.enabled=false \
  output.directory=outputs/openvla_libero10_smoke
```

联通成功的关键不是 `success_rate`，而是：

```text
policy_query_count: 1
round_trip_latency.count: 1
```

只跑一步通常无法完成任务，所以 `success_rate: 0.0` 正常。如果
`policy_query_count: 0`，查看远端 traceback 和本地
`outputs/openvla_libero10_smoke/episodes.jsonl` 中的 `termination_reason`。

## 7. 跑一个完整 LIBERO-10 task

冒烟通过后使用新的输出目录跑完整回合：

```bash
python -m libero.evaluation.eval \
  policy=openvla \
  policy.connection.base_url=http://127.0.0.1:8000 \
  policy.inference.unnorm_key=libero_10 \
  policy.adapter.image_preprocess=official_libero \
  policy.adapter.center_crop=true \
  benchmark.evaluation_config_path=evaluation_config.yaml \
  benchmark.task_ids='[0]' \
  benchmark.episodes_per_task=1 \
  rollout.execute_horizon=1 \
  rollout.max_steps=520 \
  recording.enabled=true \
  live_preview.enabled=false \
  output.directory=outputs/openvla_finetuned_libero10_task0
```

不同原始 LIBERO task suite 常用的单回合上限：

| suite | `rollout.max_steps` |
|---|---:|
| `libero_spatial` | 220 |
| `libero_object` | 280 |
| `libero_goal` | 300 |
| `libero_10` | 520 |
| `libero_90` | 400 |

切换 checkpoint 时，修改 `evaluation_config.yaml` 中的 `task_suite_name`，并让
`unnorm_key` 匹配该 checkpoint。例如 LIBERO-Object：

```yaml
task_suite_name: libero_object
```

然后在完整 evaluator 命令中使用：

```bash
policy.inference.unnorm_key=libero_object \
rollout.max_steps=280
```

不要继续使用 `libero_10`，也不要对微调模型的前六维动作再次做 OSC 放大。

## 8. 输出文件

以 `outputs/openvla_finetuned_libero10_task0` 为例：

```text
config.yaml              本次最终生效的完整配置
metadata.json            Client、运行环境和路径信息
episodes.jsonl           每个 episode 的结果与 termination_reason
summary.json              汇总成功率和延迟
videos/*.mp4              主相机录像
videos/*_wrist.mp4        腕部相机录像
action_traces/*.jsonl     每一步原始动作、最终动作和实际末端位移
```

相同目录下的 `episodes.jsonl` 会追加记录。重复实验时使用新的
`output.directory`，避免失败的旧记录和新结果混在一起。

从 WSL 打开结果目录：

```bash
explorer.exe "$(wslpath -w outputs/openvla_finetuned_libero10_task0)"
```

## 9. 不连接真实模型的本地全链路测试

新版 evaluator 自带 OpenPI 协议 mock，可验证：

```text
任务加载 → MuJoCo → 原始观测 → PolicyClient → 动作 → env.step → 录像/统计
```

终端 A：

```bash
cd /mnt/d/BenchmarkTest/LIBERO-PRO-trial
conda activate libero-pro-cpu
python -m libero.evaluation.mock_server --mode noop --chunk-size 16
```

终端 B：

```bash
cd /mnt/d/BenchmarkTest/LIBERO-PRO-trial
conda activate libero-pro-cpu
python -m libero.evaluation.eval \
  policy=mock \
  benchmark.evaluation_config_path=evaluation_config.yaml \
  benchmark.task_ids='[0]' \
  benchmark.episodes_per_task=1 \
  rollout.max_steps=2 \
  recording.enabled=false \
  live_preview.enabled=false \
  output.directory=outputs/mock_smoke
```

mock 不会完成任务，`success_rate: 0.0` 正常。它使用 OpenPI WebSocket 协议，不会验证
OpenVLA 官方 `/act` 的序列化；OpenVLA `/act` 必须用第 6 节的一步测试验证。

## 10. 常见问题

### `/openapi.json` 成功，但第一次 `/act` 失败

`/openapi.json` 只检查路由。若远端提示 `unnorm_key` 不存在，直接查看 traceback 中的
`dict_keys(...)`。微调 LIBERO-10 应包含 `libero_10`；只出现 Bridge/Open-X 键说明远端
加载了基础统计量。不要在本地改回 `bridge_orig` 来掩盖问题。

### 远端返回 HTTP 200，但本地仍报推理失败

官方 `deploy.py` 捕获异常后通常返回 JSON 字符串 `"error"`，HTTP 状态仍可能是 200。
真正原因在远端 traceback。

### `policy_query_count: 0`

第一条推理请求没有产生合法动作，episode 在执行 policy action 前终止。查看：

```bash
cat outputs/<本次目录>/episodes.jsonl
```

### 没找到视频

视频位于：

```text
outputs/<本次目录>/videos/task_<id>_episode_<id>_init_<id>.mp4
```

必须设置 `recording.enabled=true`。一步冒烟测试默认建议关闭录像。

### EGL 或 MuJoCo 初始化失败

本项目已验证的 WSL 默认后端是 EGL：

```bash
environment.render_backend=egl
```

不要在系统没有安装 OSMesa 时强制选择 `osmesa`。

## 11. 当前相关启动入口

| 入口 | 定位 | 是否用于当前远程 VLA 评测 |
|---|---|---|
| `python -m libero.evaluation.eval` | 新版统一 evaluator，选择任务、Client、rollout、录像和统计 | **推荐** |
| `python -m libero.evaluation.mock_server` | 新版 OpenPI 协议 mock server | 用于本地联通测试 |
| `policy_eval/eval_one_task.py` | 旧版单任务 HTTP Adapter evaluator | 已由远端主分支删除，不能再启动 |
| `policy_eval/mock_policy_server.py` | 旧版 HTTP mock server | 已由远端主分支删除，不能再启动 |
| `python -m libero.lifelong.main` | 原生 LIBERO 持续学习训练入口 | 不是远程 VLA evaluator |
| `python -m libero.lifelong.evaluate` | 原生 LIBERO 已训练 policy checkpoint 评测 | 不是当前两进程方案 |
| `python benchmark_scripts/render_single_task.py` | 只加载并渲染单个任务 | 环境/任务数据检查，不查询 VLA |
| `python benchmark_scripts/check_task_suites.py` | 检查 task suite、BDDL 和 init state | 数据检查，不执行 policy |
