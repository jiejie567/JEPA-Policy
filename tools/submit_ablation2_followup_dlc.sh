#!/usr/bin/env bash
set -euo pipefail

REPO="/mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy"
IMAGE="dsw-registry-vpc.cn-hangzhou.cr.aliyuncs.com/pai/training-xpu-pytorch:2.0.0-torch2.6.0-ubuntu24.04-cuda12.6-py312-01ebc85b-1e0446d7"
RESOURCE_ID="quota5wghsav7ur7"
WORKSPACE_ID="594560"

command -v aliyun >/dev/null 2>&1 || { echo "aliyun CLI is unavailable" >&2; exit 1; }

submit_job() {
  local benchmark="$1"
  local seed="$2"
  local clone_job_id="$3"
  local launcher skip_transport job_name source_job job_env job_body

  case "$benchmark" in
    libero)
      launcher="run_ablation2_libero_8.sh"
      skip_transport="0"
      ;;
    robomimic)
      launcher="run_ablation2_robomimic_12.sh"
      skip_transport="1"
      ;;
    *) echo "Unsupported benchmark: $benchmark" >&2; return 2 ;;
  esac
  job_name="${benchmark}-ablation2-seed${seed}-no-transport"

  # Reuse the source job's injected values without printing or persisting
  # secrets such as WANDB_API_KEY.
  source_job="$(aliyun pai-dlc GetJob --RegionId cn-hangzhou --JobId "$clone_job_id")"
  job_env="$(jq -c '[.CustomEnvs[]? | {(.Key): .Value}] | add // {}' <<<"$source_job")"
  [[ "$(jq -r '.WANDB_API_KEY // empty' <<<"$job_env")" != "" ]] || {
    echo "Source job $clone_job_id has no injected WANDB_API_KEY" >&2
    return 1
  }

  job_body="$(
    JOB_NAME="$job_name" BENCHMARK="$benchmark" SEED_VALUE="$seed" \
    CLONE_JOB_ID="$clone_job_id" LAUNCHER="$launcher" \
    SKIP_TRANSPORT_VALUE="$skip_transport" JOB_ENVS="$job_env" \
    IMAGE_VALUE="$IMAGE" RESOURCE_VALUE="$RESOURCE_ID" \
    WORKSPACE_VALUE="$WORKSPACE_ID" python3 - <<'PY'
import json
import os

body = {
    "DisplayName": os.environ["JOB_NAME"],
    "JobType": "PyTorchJob",
    "WorkspaceId": os.environ["WORKSPACE_VALUE"],
    "ResourceId": os.environ["RESOURCE_VALUE"],
    "Priority": 6,
    "JobMaxRunningTimeMinutes": 2880,
    "UserCommand": (
        "cd /mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy && "
        f"SEED={os.environ['SEED_VALUE']} "
        f"SKIP_TRANSPORT={os.environ['SKIP_TRANSPORT_VALUE']} "
        f"bash {os.environ['LAUNCHER']}"
    ),
    "DataSources": [
        {"DataSourceType": "OSS", "Uri": "oss://dataset-robot-hz.oss-cn-hangzhou-internal.aliyuncs.com/", "MountPath": "/mnt/oss_data/"},
        {"DataSourceType": "OSS", "Uri": "oss://models-robot-hz.oss-cn-hangzhou-internal.aliyuncs.com/", "MountPath": "/mnt/oss_models/"},
        {"DataSourceType": "OSS", "Uri": "oss://anyverse-record-hz.oss-cn-hangzhou-internal.aliyuncs.com/", "MountPath": "/mnt/record_data/"},
        {"DataSourceType": "NAS", "Uri": "nas://000l2w2byqkms68hskp-may73.cn-hangzhou.nas.aliyuncs.com/", "MountPath": "/mnt/data_nas/"},
    ],
    "UserVpc": {
        "VpcId": "vpc-bp17kz3112q1dm5xe0lnd",
        "SwitchId": "vsw-bp125xmmc1qblw0piv1z3",
        "SecurityGroupId": "sg-bp1ja5af6unqtwp2p94l",
        "DefaultRoute": "eth0",
    },
    "Tags": [
        {"Key": "CloneFromJobID", "Value": os.environ["CLONE_JOB_ID"]},
        {"Key": "SubmittedBy", "Value": "codex"},
        {"Key": "Project", "Value": "JEPA-Policy-Ablation2-Followup"},
        {"Key": "Seed", "Value": os.environ["SEED_VALUE"]},
    ],
    "Envs": json.loads(os.environ["JOB_ENVS"]),
    "JobSpecs": [{
        "Type": "Worker",
        "PodCount": 1,
        "Image": os.environ["IMAGE_VALUE"],
        "ResourceConfig": {"CPU": "160", "GPU": "16", "Memory": "1600Gi", "SharedMemory": "1600Gi"},
    }],
}
print(json.dumps(body, separators=(",", ":")))
PY
  )"

  aliyun pai-dlc CreateJob --RegionId cn-hangzhou --body "$job_body" |
    jq --arg name "$job_name" '{Name:$name,JobId,Status,RequestId}'
}

submit_job libero 41 dlc12inrpmjm9zel
submit_job robomimic 41 dlcyctuyyg16dvr9
submit_job libero 43 dlc12inrpmjm9zel
submit_job robomimic 43 dlcyctuyyg16dvr9
