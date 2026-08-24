#!/usr/bin/env bash
set -euo pipefail

REPO="/mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy"
OUTPUT="/mnt/data_nas/ykj_jepa_policy/datasets/robomimic/transport/ph/image_4cam.hdf5"
JOB_NAME="${JOB_NAME:-transport-4cam-dataset-20260722}"

[[ -x "$REPO/tools/generate_transport_4cam.sh" ]] || {
  echo "Missing generator: $REPO/tools/generate_transport_4cam.sh" >&2
  exit 1
}
[[ ! -e "$OUTPUT" ]] || {
  echo "Refusing to submit because output already exists: $OUTPUT" >&2
  exit 1
}
command -v aliyun >/dev/null 2>&1 || {
  echo "aliyun CLI is unavailable" >&2
  exit 1
}

JOB_BODY="$(JOB_NAME="$JOB_NAME" python3 - <<'PY'
import json
import os

body = {
    "DisplayName": os.environ["JOB_NAME"],
    "JobType": "PyTorchJob",
    "WorkspaceId": "594560",
    "ResourceId": "quotaa5c3d64iror",
    "Priority": 6,
    "JobMaxRunningTimeMinutes": 1440,
    "UserCommand": (
        "bash -lc 'set -euo pipefail; "
        "mkdir -p /mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy/logs; "
        "export CUDA_VISIBLE_DEVICES=0; "
        "export JEPA_RENDER_BACKEND=egl; "
        "bash /mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy/tools/generate_transport_4cam.sh "
        "2>&1 | tee /mnt/data_nas/ykj_jepa_policy/code/JEPA-Policy/logs/generate_transport_4cam.log'"
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
        {"Key": "CloneFromJobID", "Value": "dlc12inrpmjm9zel"},
        {"Key": "SubmittedBy", "Value": "codex"},
        {"Key": "Project", "Value": "JEPA-Policy-Transport-4cam"},
    ],
    "Envs": {"NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics"},
    "JobSpecs": [
        {
            "Type": "Worker",
            "PodCount": 1,
            "Image": "dsw-registry-vpc.cn-hangzhou.cr.aliyuncs.com/pai/training-xpu-pytorch:2.0.0-torch2.6.0-ubuntu24.04-cuda12.6-py312-01ebc85b-1e0446d7",
            "ResourceConfig": {
                "CPU": "10",
                "GPU": "1",
                "Memory": "96Gi",
                "SharedMemory": "96Gi",
            },
        }
    ],
}
print(json.dumps(body, separators=(",", ":")))
PY
)"

aliyun pai-dlc CreateJob --RegionId cn-hangzhou --body "$JOB_BODY"
