#!/usr/bin/env bash
set -euo pipefail

REPO="/mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy"
LAUNCHER="$REPO/run_transport_4cam_ablation2_dlc_node.sh"
SOURCE_JOB_ID="${SOURCE_JOB_ID:-dlc12inrpmjm9zel}"
IMAGE="dsw-registry-vpc.cn-hangzhou.cr.aliyuncs.com/pai/training-xpu-pytorch:2.0.0-torch2.6.0-ubuntu24.04-cuda12.6-py312-01ebc85b-1e0446d7"
RESOURCE_ID="${RESOURCE_ID:-quotaa5c3d64iror}"
WORKSPACE_ID="${WORKSPACE_ID:-594560}"
RUN_DATE="${RUN_DATE:-$(date -u +%Y%m%d)}"
RUN_SUFFIX="${RUN_SUFFIX:-formal1}"
RUN_TAG="${RUN_TAG:-}"
NODES="${NODES:-1 2 3}"
DRY_RUN="${DRY_RUN:-0}"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

command -v aliyun >/dev/null || fail "aliyun CLI is unavailable"
command -v jq >/dev/null || fail "jq is unavailable"
[[ -x "$LAUNCHER" ]] || fail "node launcher is not executable: $LAUNCHER"
[[ "$RUN_DATE" =~ ^[0-9]{8}$ ]] || fail "RUN_DATE must use YYYYMMDD"
[[ "$RUN_SUFFIX" =~ ^[A-Za-z0-9._-]+$ ]] ||
  fail "RUN_SUFFIX contains unsupported characters"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9._-]*$ ]] ||
  fail "RUN_TAG contains unsupported characters"
[[ "$DRY_RUN" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"

read -r -a requested_nodes <<<"$NODES"
(( ${#requested_nodes[@]} > 0 )) || fail "NODES must contain 1, 2, and/or 3"
declare -A seen_nodes=()
for node in "${requested_nodes[@]}"; do
  [[ "$node" == "1" || "$node" == "2" || "$node" == "3" ]] ||
    fail "NODES must contain only 1, 2, and/or 3"
  [[ -z "${seen_nodes[$node]:-}" ]] ||
    fail "NODES contains duplicate node $node"
  seen_nodes[$node]=1
done

if (( DRY_RUN != 0 )); then
  for node in "${requested_nodes[@]}"; do
    job_name="transport-4cam-ablation2-node${node}-${RUN_DATE}-${RUN_SUFFIX}"
    printf 'PLAN name=%s node=%s workspace=%s resource=%s gpus=8 cpus=80 memory=800Gi shared_memory=800Gi timeout_minutes=4320\n' \
      "$job_name" "$node" "$WORKSPACE_ID" "$RESOURCE_ID"
    printf 'COMMAND cd %q && START_GAP=180 RUN_TAG=%q bash %q %s\n' \
      "$REPO" "$RUN_TAG" "$LAUNCHER" "$node"
  done
  echo "DRY_RUN_OK no DLC job was submitted"
  exit 0
fi

source_job="$(aliyun pai-dlc GetJob \
  --RegionId cn-hangzhou \
  --JobId "$SOURCE_JOB_ID")"
job_env="$(jq -c \
  '[.CustomEnvs[]? | {(.Key): .Value}] | add // {}' <<<"$source_job")"
[[ "$(jq -r '.WANDB_API_KEY // empty' <<<"$job_env")" != "" ]] ||
  fail "source job has no injected WANDB_API_KEY"

jobs_before="$(aliyun pai-dlc ListJobs \
  --RegionId cn-hangzhou \
  --PageNumber 1 \
  --PageSize 100)"

submit_node() {
  local node="$1"
  local job_name="transport-4cam-ablation2-node${node}-${RUN_DATE}-${RUN_SUFFIX}"
  local duplicate_count body

  duplicate_count="$(jq \
    --arg name "$job_name" \
    '[.Jobs[]? | select(.DisplayName == $name)] | length' \
    <<<"$jobs_before")"
  (( duplicate_count == 0 )) ||
    fail "a DLC job already uses display name: $job_name"

  body="$(
    NODE_VALUE="$node" \
    JOB_NAME="$job_name" \
    SOURCE_JOB_ID_VALUE="$SOURCE_JOB_ID" \
    JOB_ENVS="$job_env" \
    IMAGE_VALUE="$IMAGE" \
    RESOURCE_VALUE="$RESOURCE_ID" \
    WORKSPACE_VALUE="$WORKSPACE_ID" \
    RUN_TAG_VALUE="$RUN_TAG" \
    python3 - <<'PY'
import json
import os

body = {
    "DisplayName": os.environ["JOB_NAME"],
    "JobType": "PyTorchJob",
    "WorkspaceId": os.environ["WORKSPACE_VALUE"],
    "ResourceId": os.environ["RESOURCE_VALUE"],
    "Priority": 6,
    "JobMaxRunningTimeMinutes": 4320,
    "UserCommand": (
        "cd /mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy && "
        "START_GAP=180 RUN_TAG=" + os.environ["RUN_TAG_VALUE"] + " "
        "bash run_transport_4cam_ablation2_dlc_node.sh "
        + os.environ["NODE_VALUE"]
    ),
    "DataSources": [
        {
            "DataSourceType": "OSS",
            "Uri": "oss://dataset-robot-hz.oss-cn-hangzhou-internal.aliyuncs.com/",
            "MountPath": "/mnt/oss_data/",
        },
        {
            "DataSourceType": "OSS",
            "Uri": "oss://models-robot-hz.oss-cn-hangzhou-internal.aliyuncs.com/",
            "MountPath": "/mnt/oss_models/",
        },
        {
            "DataSourceType": "OSS",
            "Uri": "oss://anyverse-record-hz.oss-cn-hangzhou-internal.aliyuncs.com/",
            "MountPath": "/mnt/record_data/",
        },
        {
            "DataSourceType": "NAS",
            "Uri": "nas://000l2w2byqkms68hskp-may73.cn-hangzhou.nas.aliyuncs.com/",
            "MountPath": "/mnt/data_nas/",
        },
    ],
    "UserVpc": {
        "VpcId": "vpc-bp17kz3112q1dm5xe0lnd",
        "SwitchId": "vsw-bp125xmmc1qblw0piv1z3",
        "SecurityGroupId": "sg-bp1ja5af6unqtwp2p94l",
        "DefaultRoute": "eth0",
    },
    "Tags": [
        {
            "Key": "CloneFromJobID",
            "Value": os.environ["SOURCE_JOB_ID_VALUE"],
        },
        {"Key": "SubmittedBy", "Value": "codex"},
        {"Key": "Project", "Value": "JEPA-Policy-Transport-4cam-Ablation2"},
        {"Key": "Node", "Value": os.environ["NODE_VALUE"]},
    ],
    "Envs": json.loads(os.environ["JOB_ENVS"]),
    "JobSpecs": [
        {
            "Type": "Worker",
            "PodCount": 1,
            "Image": os.environ["IMAGE_VALUE"],
            "ResourceConfig": {
                "CPU": "80",
                "GPU": "8",
                "Memory": "800Gi",
                "SharedMemory": "800Gi",
            },
        }
    ],
}
print(json.dumps(body, separators=(",", ":")))
PY
  )"

  aliyun pai-dlc CreateJob \
    --RegionId cn-hangzhou \
    --body "$body" |
    jq --arg name "$job_name" '{Name:$name,JobId,Status,RequestId}'
}

for node in "${requested_nodes[@]}"; do
  submit_node "$node"
done
