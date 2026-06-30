#!/usr/bin/env bash
# ============================================================
# MiniMind 训练 - AutoDL 一键引导脚本（支持多阶段）
# 用法：
#   bash run_autodl.sh pretrain        # 预训练（默认）
#   bash run_autodl.sh sft             # 指令微调（基于 pretrain 权重）
#   bash run_autodl.sh sft wandb       # 同时开启 swanlab 网页实时曲线
# 作用：克隆/更新代码 -> 装依赖 -> 下对应数据 -> 后台启动训练（实时日志）
# 可重复执行：已存在的代码/数据会跳过
# 注：开 wandb 需先在本机执行过 `swanlab login` 并粘贴 swanlab.cn 的 API Key
# ============================================================
set -e

STAGE="${1:-pretrain}"   # 第一个参数指定阶段，默认 pretrain
WANDB_FLAG=""            # 第二个参数为 wandb 时开启 swanlab 上报
[ "${2:-}" = "wandb" ] && WANDB_FLAG="--use_wandb"

# ---------- 可调参数 ----------
WORKDIR="/root/autodl-tmp"
REPO="https://github.com/hanyi-0918/financial-model-training.git"
PROJ="$WORKDIR/minimind"
BATCH_SIZE=64
MAX_SEQ_LEN=768
ACCUM_STEPS=8
EPOCHS=1
NUM_WORKERS=8
# ------------------------------

# 按阶段映射：数据文件 / 训练脚本 / 日志名
case "$STAGE" in
  pretrain)
    DATA_FILE="pretrain_t2t_mini.jsonl"; TRAIN_PY="train_pretrain.py"; LOG="pretrain.log" ;;
  sft)
    DATA_FILE="sft_t2t_mini.jsonl";      TRAIN_PY="train_full_sft.py"; LOG="full_sft.log" ;;
  *)
    echo "未知阶段：$STAGE（可选 pretrain / sft）"; exit 1 ;;
esac
DATA="$PROJ/dataset/$DATA_FILE"

echo ">>> 阶段：$STAGE | 数据：$DATA_FILE | 脚本：$TRAIN_PY"

echo ">>> [0/5] 开启 AutoDL 学术加速"
source /etc/network_turbo 2>/dev/null || echo "    (非 AutoDL 环境，跳过)"

echo ">>> [1/5] 准备代码"
mkdir -p "$WORKDIR"
if [ -d "$PROJ/.git" ]; then
  git -C "$PROJ" pull --ff-only || true
else
  git clone "$REPO" "$PROJ"
fi
cd "$PROJ"

echo ">>> [2/5] 安装依赖"
pip install -q -r requirements.txt
pip install -q modelscope

echo ">>> [3/5] 准备数据"
if [ -f "$DATA" ]; then
  echo "    数据已存在，跳过下载：$DATA"
else
  modelscope download --dataset gongjy/minimind_dataset "$DATA_FILE" --local_dir ./dataset
fi

# SFT 需要预训练权重作为起点
if [ "$STAGE" = "sft" ] && [ ! -f "$PROJ/out/pretrain_768.pth" ]; then
  echo "！！ SFT 需要 out/pretrain_768.pth 作为起点，但未找到。"
  echo "    请先跑预训练，或把本地 pretrain_768.pth 上传到 $PROJ/out/"
  exit 1
fi

echo ">>> [4/5] 检查 GPU"
python -c "import torch; print('    CUDA:', torch.cuda.is_available(), '| 设备:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo ">>> [5/5] 后台启动训练（日志写入 out/$LOG）"
mkdir -p out
cd trainer
# 注意 python -u：关闭输出缓冲，loss 实时写入日志（踩过的坑）
nohup python -u "$TRAIN_PY" \
  --device cuda:0 \
  --dtype bfloat16 \
  --batch_size $BATCH_SIZE \
  --max_seq_len $MAX_SEQ_LEN \
  --num_workers $NUM_WORKERS \
  --accumulation_steps $ACCUM_STEPS \
  --epochs $EPOCHS \
  --save_interval 1000 \
  --log_interval 100 \
  $WANDB_FLAG \
  > ../out/$LOG 2>&1 &

echo ""
echo "============================================================"
echo "阶段 [$STAGE] 训练已在后台启动！PID=$!"
echo "实时看 loss：  tail -f $PROJ/out/$LOG"
echo "查看进程：    ps aux | grep $TRAIN_PY"
echo "⚠️  训练完记得去 AutoDL 控制台【关机】，否则持续计费！"
echo "============================================================"
