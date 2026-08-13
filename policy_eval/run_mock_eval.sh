#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
mock_host="${MOCK_HOST:-127.0.0.1}"
mock_port="${MOCK_PORT:-8000}"
run_name="${RUN_NAME:-mock_noop_$(date +%Y%m%d_%H%M%S)}"
runtime_dir="${project_root}/.runtime/mock_eval"
server_log="${runtime_dir}/${run_name}_server.log"

mkdir -p "${runtime_dir}"
cd "${project_root}"

"${python_bin}" -m policy_eval.mock_policy_server \
  --host "${mock_host}" \
  --port "${mock_port}" \
  --mode noop \
  --action-horizon 5 \
  >"${server_log}" 2>&1 &
server_pid=$!

cleanup() {
  if kill -0 "${server_pid}" 2>/dev/null; then
    kill -TERM "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ready=0
for _ in $(seq 1 50); do
  if "${python_bin}" -c \
    "import urllib.request; urllib.request.urlopen('http://${mock_host}:${mock_port}/healthz', timeout=1).read()" \
    >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 0.1
done

if [[ "${ready}" != "1" ]]; then
  echo "mock policy did not become ready; see ${server_log}" >&2
  exit 2
fi

"${python_bin}" -m policy_eval.eval_one_task \
  --repo-root "${project_root}" \
  --policy-url "http://${mock_host}:${mock_port}" \
  --suite libero_object \
  --task-name pick_up_the_cream_cheese_and_place_it_in_the_basket \
  --n-episodes 1 \
  --init-state-ids 0 \
  --run-name "${run_name}" \
  "$@"
