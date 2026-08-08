# 使用 dataset-8Hz 微调 ForceVLA

## 1. 进入环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_data/envs/forcevla

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_LEROBOT_HOME=/root/autodl-tmp/lerobot_data
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi_cache
export OMP_NUM_THREADS=8

cd /root/autodl-tmp/ForceVLA
mkdir -p /root/autodl-tmp/ForceVLA/checkpoints
```

## 2. 已添加的数据配置

主配置：

```text
forcevla_8hz_all_lora
```

包含 4 个 LeRobot 2.1 数据集：

```text
/root/autodl-tmp/dataset-8Hz/flip_box
/root/autodl-tmp/dataset-8Hz/insert_plug
/root/autodl-tmp/dataset-8Hz/press_button
/root/autodl-tmp/dataset-8Hz/wipe_board
```

单任务排查配置：

```text
forcevla_8hz_flip_box_lora
forcevla_8hz_insert_plug_lora
forcevla_8hz_press_button_lora
forcevla_8hz_wipe_board_lora
```

## 3. 字段映射

```text
observation.images.third_view       -> base_0_rgb
observation.images.wrist            -> left_wrist_0_rgb
无右腕相机                          -> right_wrist_0_rgb 为 0，mask=False
observation.state[8]                -> 本体状态
observation.wrench_compensated[6]   -> 力/力矩
action[8]                           -> 7 关节动作 + 夹爪动作
```

模型输入里会把 `observation.state[8]` 和 `observation.wrench_compensated[6]` 拼成 14 维，再 padding 到 ForceVLA 默认 `action_dim=32`：

```text
state[0:8]   = 7 关节 + 夹爪本体状态
state[8:14]  = 6 维力/力矩
state[14:32] = padding
```

动作使用：

```text
action_horizon = 16
action_dim     = 32
前 8 维         = 数据集 action
后 24 维        = padding
delta mask     = 前 7 个关节转 delta，夹爪保持 absolute
```

`dataset-8Hz` 是 8Hz，`action_horizon=16` 对应约 2 秒动作 chunk。

## 4. 计算归一化统计

全量多任务：

```bash
python scripts/compute_norm_stats.py --config-name forcevla_8hz_all_lora
```

快速试跑：

```bash
python scripts/compute_norm_stats.py --config-name forcevla_8hz_all_lora --max-frames 2048
```

输出位置：

```text
/root/autodl-tmp/ForceVLA/assets/forcevla_8hz_all_lora/dataset_8hz_all
```

单任务示例：

```bash
python scripts/compute_norm_stats.py --config-name forcevla_8hz_flip_box_lora
```

## 5. 开始训练

推荐先关掉 wandb 做本地训练：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py forcevla_8hz_all_lora \
  --exp-name dataset_8hz_all_lora_v1 \
  --checkpoint-base-dir /root/autodl-tmp/ForceVLA/checkpoints \
  --batch-size 16 \
  --num-train-steps 30000 \
  --save-interval 2000 \
  --keep-period 10000 \
  --no-wandb-enabled \
  --overwrite
```

如果需要 wandb：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py forcevla_8hz_all_lora \
  --exp-name dataset_8hz_all_lora_v1 \
  --checkpoint-base-dir /root/autodl-tmp/ForceVLA/checkpoints \
  --batch-size 16 \
  --num-train-steps 30000 \
  --save-interval 2000 \
  --keep-period 10000 \
  --wandb-enabled \
  --overwrite
```

权重保存位置：

```text
/root/autodl-tmp/ForceVLA/checkpoints/forcevla_8hz_all_lora/dataset_8hz_all_lora_v1/<step>/params
```

## 6. 断点续训

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py forcevla_8hz_all_lora \
  --exp-name dataset_8hz_all_lora_v1 \
  --checkpoint-base-dir /root/autodl-tmp/ForceVLA/checkpoints \
  --batch-size 16 \
  --num-train-steps 30000 \
  --save-interval 2000 \
  --keep-period 10000 \
  --no-wandb-enabled \
  --resume
```

## 7. 常用参数含义

```text
--exp-name
  实验名，也是 checkpoint 子目录名。

--checkpoint-base-dir
  checkpoint 根目录；本任务固定为 /root/autodl-tmp/ForceVLA/checkpoints。

--batch-size
  全局 batch size。LoRA 默认 16；显存不够可改 8 或 4。

--num-train-steps
  总训练 step 数。多任务先用 30000；单任务可先用 10000。

--save-interval
  每隔多少 step 保存一次。

--keep-period
  满足 step % keep_period == 0 的 checkpoint 会长期保留。

--overwrite
  删除同名实验旧 checkpoint 后重新训练。

--resume
  从同名实验最后一个 checkpoint 继续训练，不能和 --overwrite 同时用。

--no-wandb-enabled
  关闭 wandb，适合先做本地验证。
```

## 8. 单任务训练示例

```bash
python scripts/compute_norm_stats.py --config-name forcevla_8hz_insert_plug_lora

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python scripts/train.py forcevla_8hz_insert_plug_lora \
  --exp-name insert_plug_lora_v1 \
  --checkpoint-base-dir /root/autodl-tmp/ForceVLA/checkpoints \
  --batch-size 4 \
  --num-train-steps 10000 \
  --save-interval 1000 \
  --keep-period 5000 \
  --no-wandb-enabled \
  --overwrite
```

## 9. 复查命令

查看配置是否注册：

```bash
python scripts/train.py --help
```

查看某个配置参数：

```bash
python scripts/train.py forcevla_8hz_all_lora --help
```

检查 checkpoint：

```bash
find /root/autodl-tmp/ForceVLA/checkpoints/forcevla_8hz_all_lora/dataset_8hz_all_lora_v1 -maxdepth 2 -type d | sort
```
