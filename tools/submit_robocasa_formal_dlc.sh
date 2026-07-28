#!/usr/bin/env bash
set -euo pipefail

REPO="/mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy"
SOURCE_JOB_ID="${SOURCE_JOB_ID:-dlc12inrpmjm9zel}"
IMAGE="dsw-registry-vpc.cn-hangzhou.cr.aliyuncs.com/pai/training-xpu-pytorch:2.0.0-torch2.6.0-ubuntu24.04-cuda12.6-py312-01ebc85b-1e0446d7"
RESOURCE_ID="${RESOURCE_ID:-quotaa5c3d64iror}"
RUN_SUFFIX="${RUN_SUFFIX:-formal1}"
RUN_TAG="${RUN_TAG:-$RUN_SUFFIX}"
RUN_DATE="${RUN_DATE:-$(date -u +%Y%m%d)}"
NODES="${NODES:-1 2 3}"
NODE_GPUS="${NODE_GPUS:-16}"
NODE_CPUS="${NODE_CPUS:-160}"
NODE_MEMORY="${NODE_MEMORY:-1600Gi}"
NODE_SHARED_MEMORY="${NODE_SHARED_MEMORY:-1600Gi}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RESUME_EXISTING="${RESUME_EXISTING:-1}"

command -v aliyun >/dev/null || { echo "aliyun CLI unavailable" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq unavailable" >&2; exit 1; }
[[ "$RUN_DATE" =~ ^[0-9]{8}$ ]] || {
  echo "RUN_DATE must use YYYYMMDD" >&2
  exit 1
}
[[ "$RUN_SUFFIX" =~ ^[A-Za-z0-9._-]+$ &&
  "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "RUN_SUFFIX and RUN_TAG contain unsupported characters" >&2
  exit 1
}
[[ "$AUTO_RESUME" =~ ^[01]$ && "$RESUME_EXISTING" =~ ^[01]$ ]] || {
  echo "AUTO_RESUME and RESUME_EXISTING must be 0 or 1" >&2
  exit 1
}
read -r -a requested_nodes <<<"$NODES"
(( ${#requested_nodes[@]} > 0 )) || {
  echo "NODES must contain node 1, 2, and/or 3" >&2
  exit 1
}
declare -A seen_nodes=()
for node in "${requested_nodes[@]}"; do
  [[ "$node" == "1" || "$node" == "2" || "$node" == "3" ]] || {
    echo "NODES must contain only 1, 2, and/or 3" >&2
    exit 1
  }
  [[ -z "${seen_nodes[$node]:-}" ]] || {
    echo "NODES contains duplicate node $node" >&2
    exit 1
  }
  seen_nodes[$node]=1
done
[[ -x "$REPO/run_robocasa_dlc_node.sh" ]] || {
  echo "RoboCasa node launcher is not executable" >&2
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
  local job_name="robocasa-formal-node${node}-${RUN_DATE}-${RUN_SUFFIX}"
  local body
  body="$(
    NODE_VALUE="$node" JOB_NAME="$job_name" SOURCE_JOB_ID="$SOURCE_JOB_ID" \
    JOB_ENVS="$job_env" IMAGE_VALUE="$IMAGE" RESOURCE_VALUE="$RESOURCE_ID" \
    RUN_TAG_VALUE="$RUN_TAG" NODE_GPUS_VALUE="$NODE_GPUS" \
    NODE_CPUS_VALUE="$NODE_CPUS" NODE_MEMORY_VALUE="$NODE_MEMORY" \
    NODE_SHARED_MEMORY_VALUE="$NODE_SHARED_MEMORY" \
    AUTO_RESUME_VALUE="$AUTO_RESUME" RESUME_EXISTING_VALUE="$RESUME_EXISTING" \
    python3 - <<'PY'
import json, os
body = {
    "DisplayName": os.environ["JOB_NAME"],
    "JobType": "PyTorchJob",
    "WorkspaceId": "594560",
    "ResourceId": os.environ["RESOURCE_VALUE"],
    "Priority": 6,
    "JobMaxRunningTimeMinutes": 4320,
    "UserCommand": (
        "cd /mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy && "
        "ROBOCASA_WANDB_MODE=online SYNC_WANDB=0 "
        "START_GAP=180 AUTO_RESUME=" + os.environ["AUTO_RESUME_VALUE"] + " "
        "RESUME_EXISTING=" + os.environ["RESUME_EXISTING_VALUE"] + " "
        "EVAL_FREQ=10000 VALIDATION_FREQ=10000 EVAL_WORKERS=10 "
        "RUN_TAG=" + os.environ["RUN_TAG_VALUE"] +
        " bash run_robocasa_dlc_node.sh " + os.environ["NODE_VALUE"]
    ),
    "DataSources": [
        {"DataSourceType":"OSS","Uri":"oss://dataset-robot-hz.oss-cn-hangzhou-internal.aliyuncs.com/","MountPath":"/mnt/oss_data/"},
        {"DataSourceType":"OSS","Uri":"oss://models-robot-hz.oss-cn-hangzhou-internal.aliyuncs.com/","MountPath":"/mnt/oss_models/"},
        {"DataSourceType":"OSS","Uri":"oss://anyverse-record-hz.oss-cn-hangzhou-internal.aliyuncs.com/","MountPath":"/mnt/record_data/"},
        {"DataSourceType":"NAS","Uri":"nas://000l2w2byqkms68hskp-may73.cn-hangzhou.nas.aliyuncs.com/","MountPath":"/mnt/data_nas/"},
    ],
    "UserVpc": {
        "VpcId":"vpc-bp17kz3112q1dm5xe0lnd",
        "SwitchId":"vsw-bp125xmmc1qblw0piv1z3",
        "SecurityGroupId":"sg-bp1ja5af6unqtwp2p94l",
        "DefaultRoute":"eth0",
    },
    "Tags": [
        {"Key":"CloneFromJobID","Value":os.environ["SOURCE_JOB_ID"]},
        {"Key":"SubmittedBy","Value":"codex"},
        {"Key":"Project","Value":"JEPA-Policy-RoboCasa-Formal"},
        {"Key":"Node","Value":os.environ["NODE_VALUE"]},
    ],
    "Envs": json.loads(os.environ["JOB_ENVS"]),
    "JobSpecs": [{
        "Type":"Worker",
        "PodCount":1,
        "Image":os.environ["IMAGE_VALUE"],
        "ResourceConfig":{
            "CPU":os.environ["NODE_CPUS_VALUE"],
            "GPU":os.environ["NODE_GPUS_VALUE"],
            "Memory":os.environ["NODE_MEMORY_VALUE"],
            "SharedMemory":os.environ["NODE_SHARED_MEMORY_VALUE"],
        },
    }],
}
print(json.dumps(body, separators=(",", ":")))
PY
  )"
  aliyun pai-dlc CreateJob --RegionId cn-hangzhou --body "$body" |
    jq --arg name "$job_name" '{Name:$name,JobId,Status,RequestId}'
}

for node in "${requested_nodes[@]}"; do
  submit_node "$node"
done
