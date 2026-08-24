#!/usr/bin/env bash
set -euo pipefail

REPO="/mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy"
SOURCE_JOB_ID="${SOURCE_JOB_ID:-dlc12inrpmjm9zel}"
RESOURCE_MODE="${RESOURCE_MODE:-ecs}"
RESOURCE_ID="${RESOURCE_ID:-quotaa5c3d64iror}"
ECS_IMAGE="dsw-registry-vpc.cn-hangzhou.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312"
LINGJUN_IMAGE="dsw-registry-vpc.cn-hangzhou.cr.aliyuncs.com/pai/training-xpu-pytorch:2.0.0-torch2.6.0-ubuntu24.04-cuda12.6-py312-01ebc85b-1e0446d7"
ECS_SPEC="${ECS_SPEC:-ecs.gn8is-2x.8xlarge}"
SPOT_DISCOUNT_LIMIT="${SPOT_DISCOUNT_LIMIT:-1.0}"
TRAIN_GPU_COUNT="${TRAIN_GPU_COUNT:-2}"
RUNS_PER_GPU="${RUNS_PER_GPU:-2}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-6}"
START_GAP="${START_GAP:-120}"
RUN_SUFFIX="${RUN_SUFFIX:-formal1}"
RUN_TAG="${RUN_TAG:-$RUN_SUFFIX}"
RUN_DATE="${RUN_DATE:-$(date -u +%Y%m%d)}"
NODES="${NODES:-1}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
ALLOW_CONCURRENT_NODES="${ALLOW_CONCURRENT_NODES:-0}"
JOB_MAX_RUNNING_TIME_MINUTES="${JOB_MAX_RUNNING_TIME_MINUTES:-20160}"

command -v aliyun >/dev/null || { echo "aliyun CLI unavailable" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq unavailable" >&2; exit 1; }
[[ "$RESOURCE_MODE" == "ecs" || "$RESOURCE_MODE" == "lingjun" ]] || {
  echo "RESOURCE_MODE must be ecs or lingjun" >&2
  exit 1
}
if [[ "$RESOURCE_MODE" == "lingjun" ]]; then
  [[ -n "$RESOURCE_ID" ]] || {
    echo "RESOURCE_ID is required for Lingjun jobs" >&2
    exit 1
  }
  IMAGE="$LINGJUN_IMAGE"
else
  IMAGE="$ECS_IMAGE"
fi
[[ "$RUN_DATE" =~ ^[0-9]{8}$ ]] || {
  echo "RUN_DATE must use YYYYMMDD" >&2
  exit 1
}
[[ "$SPOT_DISCOUNT_LIMIT" =~ ^0(\.[0-9]+)?$|^1(\.0+)?$ ]] || {
  echo "SPOT_DISCOUNT_LIMIT must be between 0 and 1" >&2
  exit 1
}
[[ "$RUN_SUFFIX" =~ ^[A-Za-z0-9._-]+$ &&
  "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "RUN_SUFFIX and RUN_TAG contain unsupported characters" >&2
  exit 1
}
[[ "$PREFLIGHT_ONLY" =~ ^[01]$ ]] || {
  echo "PREFLIGHT_ONLY must be 0 or 1" >&2
  exit 1
}
if [[ "$RESOURCE_MODE" == "lingjun" &&
  "$RESOURCE_ID" == "quotaa5c3d64iror" && "$PREFLIGHT_ONLY" == "0" ]]; then
  echo "The current ml.gp7vf PAI-PPU quota has no Vulkan rendering device; RoboTwin formal jobs require ECS or a Vulkan-capable NVIDIA quota" >&2
  exit 1
fi
[[ "$ALLOW_CONCURRENT_NODES" =~ ^[01]$ ]] || {
  echo "ALLOW_CONCURRENT_NODES must be 0 or 1" >&2
  exit 1
}
for value in \
  "$TRAIN_GPU_COUNT" "$RUNS_PER_GPU" "$DATALOADER_WORKERS" "$START_GAP" \
  "$JOB_MAX_RUNNING_TIME_MINUTES"; do
  [[ "$value" =~ ^[0-9]+$ ]] || {
    echo "GPU, worker, and launch-gap settings must be integers" >&2
    exit 1
  }
done
(( TRAIN_GPU_COUNT > 0 && RUNS_PER_GPU > 0 &&
  DATALOADER_WORKERS > 0 && START_GAP >= 0 )) || {
  echo "GPU, run, and worker counts must be positive" >&2
  exit 1
}
read -r -a requested_nodes <<<"$NODES"
(( ${#requested_nodes[@]} > 0 )) || {
  echo "NODES must contain node 1 and/or node 2" >&2
  exit 1
}
declare -A seen_nodes=()
for node in "${requested_nodes[@]}"; do
  [[ "$node" == "1" || "$node" == "2" ]] || {
    echo "NODES must contain only 1 and/or 2" >&2
    exit 1
  }
  [[ -z "${seen_nodes[$node]:-}" ]] || {
    echo "NODES contains duplicate node $node" >&2
    exit 1
  }
  seen_nodes[$node]=1
done
if [[ "$RESOURCE_MODE" == "ecs" ]] &&
  (( ${#requested_nodes[@]} > 1 && ALLOW_CONCURRENT_NODES == 0 )); then
  echo "Submit one RoboTwin node at a time under the current 2-GPU quota; set NODES=1 or NODES=2" >&2
  exit 1
fi
[[ -x "$REPO/run_robotwin_dlc_node.sh" ]] || {
  echo "RoboTwin node launcher is not executable" >&2
  exit 1
}

source_job="$(aliyun pai-dlc GetJob --RegionId cn-hangzhou --JobId "$SOURCE_JOB_ID")"
job_env="$(jq -c '[.CustomEnvs[]? | {(.Key): .Value}] | add // {}' <<<"$source_job")"
[[ "$(jq -r '.WANDB_API_KEY // empty' <<<"$job_env")" != "" ]] || {
  echo "Source job has no injected WANDB_API_KEY" >&2
  exit 1
}

submit_node() {
  local node="$1"
  local job_name="robotwin-formal-node${node}-${RUN_DATE}-${RUN_SUFFIX}"
  local body
  body="$(
    NODE_VALUE="$node" JOB_NAME="$job_name" SOURCE_JOB_ID="$SOURCE_JOB_ID" \
    JOB_ENVS="$job_env" IMAGE_VALUE="$IMAGE" ECS_SPEC_VALUE="$ECS_SPEC" \
    RESOURCE_MODE_VALUE="$RESOURCE_MODE" RESOURCE_ID_VALUE="$RESOURCE_ID" \
    SPOT_DISCOUNT_VALUE="$SPOT_DISCOUNT_LIMIT" \
    RUN_TAG_VALUE="$RUN_TAG" PREFLIGHT_ONLY_VALUE="$PREFLIGHT_ONLY" \
    TRAIN_GPU_COUNT_VALUE="$TRAIN_GPU_COUNT" \
    RUNS_PER_GPU_VALUE="$RUNS_PER_GPU" \
    DATALOADER_WORKERS_VALUE="$DATALOADER_WORKERS" \
    START_GAP_VALUE="$START_GAP" \
    JOB_MAX_RUNNING_TIME_VALUE="$JOB_MAX_RUNNING_TIME_MINUTES" \
    python3 - <<'PY'
import json, os
job_envs = json.loads(os.environ["JOB_ENVS"])
job_envs["NVIDIA_DRIVER_CAPABILITIES"] = "all"
resource_mode = os.environ["RESOURCE_MODE_VALUE"]
gpu_count = int(os.environ["TRAIN_GPU_COUNT_VALUE"])
body = {
    "DisplayName": os.environ["JOB_NAME"],
    "JobType": "PyTorchJob",
    "WorkspaceId": "594560",
    "Priority": 6,
    "JobMaxRunningTimeMinutes": int(os.environ["JOB_MAX_RUNNING_TIME_VALUE"]),
    "UserCommand": (
        "cd /mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy && "
        "ROBOTWIN_WANDB_MODE=online TRAIN_GPU_COUNT=" +
        os.environ["TRAIN_GPU_COUNT_VALUE"] + " RUNS_PER_GPU=" +
        os.environ["RUNS_PER_GPU_VALUE"] + " DATALOADER_WORKERS=" +
        os.environ["DATALOADER_WORKERS_VALUE"] + " START_GAP=" +
        os.environ["START_GAP_VALUE"] + " " +
        "PREFLIGHT_ONLY=" + os.environ["PREFLIGHT_ONLY_VALUE"] + " RUN_TAG=" +
        os.environ["RUN_TAG_VALUE"] +
        " bash run_robotwin_dlc_node.sh " + os.environ["NODE_VALUE"]
    ),
    "DataSources": [
        {"DataSourceType":"OSS","Uri":"oss://dataset-robot-hz.oss-cn-hangzhou-internal.aliyuncs.com/","MountPath":"/mnt/oss_data/"},
        {"DataSourceType":"OSS","Uri":"oss://models-robot-hz.oss-cn-hangzhou-internal.aliyuncs.com/","MountPath":"/mnt/oss_models/"},
        {"DataSourceType":"OSS","Uri":"oss://anyverse-record-hz.oss-cn-hangzhou-internal.aliyuncs.com/","MountPath":"/mnt/record_data/"},
        {"DataSourceType":"NAS","Uri":"nas://000l2w2byqkms68hskp-may73.cn-hangzhou.nas.aliyuncs.com/","MountPath":"/mnt/data_nas/"}
    ],
    "UserVpc": {
        "VpcId":"vpc-bp17kz3112q1dm5xe0lnd",
        "SwitchId":"vsw-bp125xmmc1qblw0piv1z3",
        "SecurityGroupId":"sg-bp1ja5af6unqtwp2p94l",
        "DefaultRoute":"eth0"
    },
    "Tags": [
        {"Key":"CloneFromJobID","Value":os.environ["SOURCE_JOB_ID"]},
        {"Key":"SubmittedBy","Value":"codex"},
        {"Key":"Project","Value":"JEPA-Policy-RoboTwin-Formal"},
        {"Key":"Node","Value":os.environ["NODE_VALUE"]},
        {"Key":"ResourceMode","Value":resource_mode}
    ],
    "Envs": job_envs,
}
job_spec = {
        "Type":"Worker",
        "PodCount":1,
        "Image":os.environ["IMAGE_VALUE"],
}
if resource_mode == "lingjun":
    body["ResourceId"] = os.environ["RESOURCE_ID_VALUE"]
    job_spec["ResourceConfig"] = {
        "CPU": str(gpu_count * 10),
        "GPU": str(gpu_count),
        "Memory": f"{gpu_count * 100}Gi",
        "SharedMemory": f"{gpu_count * 100}Gi",
    }
else:
    job_spec["EcsSpec"] = os.environ["ECS_SPEC_VALUE"]
    job_spec["SpotSpec"] = {
            "SpotStrategy":"SpotWithPriceLimit",
            "SpotDiscountLimit":float(os.environ["SPOT_DISCOUNT_VALUE"])
    }
body["JobSpecs"] = [job_spec]
print(json.dumps(body, separators=(",", ":")))
PY
  )"
  aliyun pai-dlc CreateJob --RegionId cn-hangzhou --body "$body" |
    jq --arg name "$job_name" '{Name:$name,JobId,Status,RequestId}'
}

for node in "${requested_nodes[@]}"; do
  submit_node "$node"
done
