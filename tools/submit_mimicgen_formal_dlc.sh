#!/usr/bin/env bash
set -euo pipefail

REPO="/mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy"
SOURCE_JOB_ID="${SOURCE_JOB_ID:-dlc12inrpmjm9zel}"
IMAGE="dsw-registry-vpc.cn-hangzhou.cr.aliyuncs.com/pai/training-xpu-pytorch:2.0.0-torch2.6.0-ubuntu24.04-cuda12.6-py312-01ebc85b-1e0446d7"
RESOURCE_ID="${RESOURCE_ID:-quotaa5c3d64iror}"
RUN_SUFFIX="${RUN_SUFFIX:-retry1}"

command -v aliyun >/dev/null || { echo "aliyun CLI unavailable" >&2; exit 1; }
command -v jq >/dev/null || { echo "jq unavailable" >&2; exit 1; }

source_job="$(aliyun pai-dlc GetJob --RegionId cn-hangzhou --JobId "$SOURCE_JOB_ID")"
job_env="$(jq -c '[.CustomEnvs[]? | {(.Key): .Value}] | add // {}' <<<"$source_job")"
[[ "$(jq -r '.WANDB_API_KEY // empty' <<<"$job_env")" != "" ]] || {
  echo "Source job has no injected WANDB_API_KEY" >&2
  exit 1
}

submit_node() {
  local node="$1"
  local job_name="mimicgen-formal-node${node}-20260724-${RUN_SUFFIX}"
  local body
  body="$(
    NODE_VALUE="$node" JOB_NAME="$job_name" SOURCE_JOB_ID="$SOURCE_JOB_ID" \
    JOB_ENVS="$job_env" IMAGE_VALUE="$IMAGE" RESOURCE_VALUE="$RESOURCE_ID" \
    python3 - <<'PY'
import json, os
body = {
    "DisplayName": os.environ["JOB_NAME"],
    "JobType": "PyTorchJob",
    "WorkspaceId": "594560",
    "ResourceId": os.environ["RESOURCE_VALUE"],
    "Priority": 6,
    "JobMaxRunningTimeMinutes": 2880,
    "UserCommand": (
        "cd /mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy && "
        "START_GAP=180 bash run_mimicgen_dlc_node.sh " + os.environ["NODE_VALUE"]
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
        {"Key":"Project","Value":"JEPA-Policy-MimicGen-Formal"},
        {"Key":"Node","Value":os.environ["NODE_VALUE"]},
    ],
    "Envs": json.loads(os.environ["JOB_ENVS"]),
    "JobSpecs": [{
        "Type":"Worker",
        "PodCount":1,
        "Image":os.environ["IMAGE_VALUE"],
        "ResourceConfig":{"CPU":"160","GPU":"16","Memory":"1600Gi","SharedMemory":"1600Gi"},
    }],
}
print(json.dumps(body, separators=(",", ":")))
PY
  )"
  aliyun pai-dlc CreateJob --RegionId cn-hangzhou --body "$body" |
    jq --arg name "$job_name" '{Name:$name,JobId,Status,RequestId}'
}

[[ -x "$REPO/run_mimicgen_dlc_node.sh" ]] || {
  echo "Launcher is not executable" >&2
  exit 1
}
submit_node 1
submit_node 2
