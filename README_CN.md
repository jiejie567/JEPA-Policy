<div align="center">

# JEPA Policy

**通过动作与未来表征的配对预测，实现无扩散过程的模仿学习**

[论文](https://arxiv.org/abs/2609.09630) · [项目主页](https://jiejie567.github.io/JEPA-Policy/) · [代码仓库](https://github.com/jiejie567/JEPA-Policy)

</div>

[English](./README.md) | 中文

JEPA Policy 训练机器人策略，使其同时预测专家动作片段，以及执行该动作后产生的未来观测的视觉表征。动作 token 与未来表征 token 共享同一个 Transformer，并通过两次前馈预测逐步细化。本实现基于 [Minimum Flow Policies](https://github.com/simchowitzlabpublic/much-ado-about-noising) 代码库。

## 方法概览

<p align="center">
  <a href="docs/images/method-comparison.jpg"><img src="docs/images/method-comparison.jpg" width="460" alt="Diffusion Policy、普通 JEPA、ACT-JEPA 与 JEPA Policy 的原理对比"></a>
</p>

图 1 对比的是设计范式，而非性能。JEPA Policy 在共享 Transformer 中，通过两步无扩散预测联合生成动作与未来表征。点击图片可查看高清原图。

## 特性

- 动作与未来表征共享 Transformer
- 两步、无扩散过程的 MIP 训练与推理
- 端到端视觉编码器，对未来目标表征停止梯度传播
- 支持图像观测和状态观测
- 支持 robomimic、LIBERO 和 MimicGen 任务
- 使用 Hydra 配置训练与评估
- 支持 JEPA Policy、MIP 和 Diffusion Policy 在 ARX5/X5 真实机器人上的推理

## 仿真结果

<p align="center">
  <a href="docs/images/simulation-radar.png"><img src="docs/images/simulation-radar.png" width="560" alt="包含 Hammer Cleanup 的九任务成功率对比"></a>
</p>

九个仿真任务的成功率对比，包含 Hammer Cleanup。曲线为三个训练种子的最佳 checkpoint 成功率均值，阴影为种子间范围。径向坐标为 40–100%，向下三角标记低于坐标下限的种子结果。完整结果与评估协议见[项目主页](https://jiejie567.github.io/JEPA-Policy/)及论文。

## 安装

JEPA Policy 需要 Python 3.12 和 PyTorch。默认环境支持 robomimic：

```bash
git clone https://github.com/jiejie567/JEPA-Policy.git
cd JEPA-Policy
uv sync --extra dev
```

无显示界面的 MuJoCo 渲染：

```bash
export MUJOCO_GL=egl
```

LIBERO 和 MimicGen 使用不同版本的 robosuite。请按照各自的官方安装说明将它们安装到独立环境中，再在每个环境中以可编辑模式安装本仓库。

## 快速开始

在 robomimic Square 任务上训练 JEPA Policy：

```bash
uv run examples/train_robomimic.py \
  -cn exps/jepa_policy.yaml \
  task=square_ph_image
```

训练对齐设置的纯动作 MIP 基线：

```bash
uv run examples/train_robomimic.py \
  -cn exps/mip_action_only.yaml \
  task=square_ph_image
```

评估模型检查点：

```bash
uv run examples/train_robomimic.py \
  -cn exps/jepa_policy.yaml \
  task=square_ph_image \
  mode=eval \
  optimization.model_path=/path/to/checkpoint.pt
```

可通过命令行覆盖任意 Hydra 配置项，例如：

```bash
uv run examples/train_robomimic.py \
  -cn exps/jepa_policy.yaml \
  task=tool_hang_ph_image \
  optimization.seed=41 \
  optimization.gradient_steps=300000
```

## 真实机器人评估

[`real_robot/`](real_robot/README.md) 目录包含 ARX5/X5 实验所使用的推理系统。JEPA Policy、对齐设置的 MIP 基线和 Diffusion Policy 均通过同一套观测、动作、安全、记录和操作员标注流程进行评估。

真实机器人发布内容包括：

- 文中实验所用的部署源码快照，以及固定版本的 ARX5 SDK；
- 启用硬件控制前的离线试运行和仅观测试运行；
- 文中全部五项任务、三种方法的启动脚本；
- 外部检查点与统计文件的目录布局，以及包含 75 个检查点的产物清单；
- 明确的硬件、相机、控制频率、标定和急停要求。

检查点、数据集、机器人运行记录、相机序列号、编译后的 SDK 二进制文件和内部机器路径均不纳入 Git。运行任何启动脚本前，请先阅读[真实机器人安装与安全指南](real_robot/README.md)。

## 论文基线

主要对比实验使用的纯动作 MIP 精确预设，以及对齐后的 Diffusion Policy 实验方案，记录在 [`baselines/`](baselines/README.md) 中。Diffusion Policy 相关材料固定了上游提交版本，仅提供复现论文所需的任务配置和少量实验协议适配文件，不重复收录第三方仓库。

## 支持的基准

仓库包含标准 robomimic 图像与状态任务的配置，以及用于验证 JEPA Policy 的图像任务配置：

- robomimic：Square、Tool Hang 和四相机 Transport
- LIBERO：MokaMoka 和 MugMug
- MimicGen：Coffee Preparation、Kitchen 和 Three Piece Assembly

robomimic 数据集通过 `examples/configs/task/robomimic_base.yaml` 中配置的数据集仓库进行解析。

对于 LIBERO，请配置安装和数据集路径：

```bash
export LIBERO_ROOT=/path/to/LIBERO
```

对于 MimicGen，请下载三个任务的数据集并设置根目录：

```bash
bash tools/download_mimicgen_core.sh
export MIMICGEN_DATA_ROOT=$PWD/datasets/mimicgen/core
```

## 方法配置

公开的 JEPA Policy 预设采用以下设置：

- robomimic/MimicGen 的动作预测长度为 10，LIBERO 为 16
- 两次 MIP 预测，插值时间为 0.9
- 一个未来表征 token
- 未来预测跨度为 4
- 自适应未来损失比例为 0.1
- 当前观测和未来观测采用共享裁剪，以保持时序一致性

任务配置文件定义各环境的预测长度、观测、动作维度和数据集位置。方法配置在不同任务之间共享。

## 测试

```bash
uv run pytest -q \
  tests/test_joint_action_future_mip.py \
  tests/test_future_loss_ratio.py \
  tests/test_temporal_consistent_crop.py
```

## 匿名审稿导出

从当前精确提交生成不含 Git 历史的审稿归档：

```bash
bash tools/export_anonymous.sh /path/to/JEPA-Policy-anonymous.tar.gz
```

设置 `ANON_REPOSITORY_URL`，可将公开克隆地址替换为匿名审稿地址。导出工具会拒绝打包含有已知作者标识或机器专属路径的内容。

## 致谢

本代码库扩展自 [Minimum Flow Policies](https://github.com/simchowitzlabpublic/much-ado-about-noising)，并使用 robomimic、robosuite、LIBERO 和 MimicGen。

## 许可证

本项目采用 MIT 许可证。第三方基准和数据集保留各自的许可证。
