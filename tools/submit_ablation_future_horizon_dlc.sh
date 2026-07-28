#!/usr/bin/env bash
set -euo pipefail

REPO="/mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy"
IMAGE="dsw-registry-vpc.cn-hangzhou.cr.aliyuncs.com/pai/training-xpu-pytorch:2.0.0-torch2.6.0-ubuntu24.04-cuda12.6-py312-01ebc85b-1e0446d7"

command -v aliyun >/dev/null 2>&1 || { echo "aliyun CLI is unavailable" >&2; exit 1; }

submit_job() {
  local benchmark="$1"
  local clone_job_id="$2"
  local job_name="${benchmark}-future-horizon-2-6-seeds-41-42-43"
  local source_job job_env job_body

  source_job="$(aliyun pai-dlc GetJob --RegionId cn-hangzhou --JobId "$clone_job_id")"
  job_env="$(jq -c '[.CustomEnvs[]? | {(.Key): .Value}] | add // {}' <<<"$source_job")"
  [[ "$(jq -r '.WANDB_API_KEY // empty' <<<"$job_env")" != "" ]] || {
    echo "Source job $clone_job_id has no injected WANDB_API_KEY" >&2
    return 1
  }

  job_body="$(
    JOB_NAME="$job_name" BENCHMARK_VALUE="$benchmark" CLONE_JOB_ID="$clone_job_id" \
    JOB_ENVS="$job_env" IMAGE_VALUE="$IMAGE" python3 - <<'PY'
import json
import os

benchmark = os.environ["BENCHMARK_VALUE"]
body = {
    "DisplayName": os.environ["JOB_NAME"],
    "JobType": "PyTorchJob",
    "WorkspaceId": "594560",
    "ResourceId": "quota5wghsav7ur7",
    "Priority": 6,
    "JobMaxRunningTimeMinutes": 2880,
    "UserCommand": (
        "cd /mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy && "
        f"bash run_ablation_future_horizon_12.sh {benchmark}"
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
        {"Key": "Project", "Value": "JEPA-Policy-Future-Horizon-Ablation"},
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

submit_job libero dlc12inrpmjm9zel
submit_job robomimic dlcyctuyyg16dvr9
