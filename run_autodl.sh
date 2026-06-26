#!/usr/bin/env bash
# ============================================================
# MiniMind 预训练 - AutoDL 一键引导脚本
# 用法：上传本文件到 AutoDL 服务器后执行
#   bash run_autodl.sh
# 作用：克隆代码 -> 下载数据 -> 装依赖 -> 后台启动预训练
# 可重复执行：已存在的代码/数据会跳过，不会重复下载
# ============================================================
set -e

# ---------- 可调参数 ----------
WORKDIR="/root/autodl-tmp"                 # AutoDL 数据盘（持久、大）
REPO="https://github.com/hanyi-0918/financial-model-training.git"
PROJ="$WORKDIR/minimind"
DATA="$PROJ/dataset/pretrain_t2t_mini.jsonl"

# 训练超参（按显存调整）
BATCH_SIZE=64
MAX_SEQ_LEN=768
ACCUM_STEPS=8
EPOCHS=1
NUM_WORKERS=8
# ------------------------------

echo ">>> [0/5] 开启 AutoDL 学术加速（GitHub/HF 提速，仅 AutoDL 有效）"
source /etc/network_turbo 2>/dev/null || echo "    (非 AutoDL 环境，跳过)"

echo ">>> [1/5] 准备代码"
mkdir -p "$WORKDIR"
if [ -d "$PROJ/.git" ]; then
  echo "    已存在仓库，git pull 更新"
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
  modelscope download --dataset gongjy/minimind_dataset \
    pretrain_t2t_mini.jsonl --local_dir ./dataset
fi

echo ">>> [4/5] 检查 GPU"
python -c "import torch; print('    CUDA:', torch.cuda.is_available(), '| 设备:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo ">>> [5/5] 后台启动预训练（日志写入 out/pretrain.log）"
mkdir -p out
cd trainer
nohup python train_pretrain.py \
  --device cuda:0 \
  --dtype bfloat16 \
  --batch_size $BATCH_SIZE \
  --max_seq_len $MAX_SEQ_LEN \
  --num_workers $NUM_WORKERS \
  --accumulation_steps $ACCUM_STEPS \
  --epochs $EPOCHS \
  --save_interval 1000 \
  --log_interval 100 \
  > ../out/pretrain.log 2>&1 &

echo ""
echo "============================================================"
echo "训练已在后台启动！PID=$!"
echo "实时看 loss：  tail -f $PROJ/out/pretrain.log"
echo "查看进程：    ps aux | grep train_pretrain"
echo "权重保存于：  $PROJ/out/pretrain_768.pth"
echo "⚠️  训练完记得去 AutoDL 控制台【关机】，否则持续计费！"
echo "============================================================"
