#!/usr/bin/env bash
# ============================================================
# MiniMind LoRA 训练（轻量版，同一台机器 DPO 已训好时直接用）
# 用法：
#   bash run_autodl_lora.sh              # 串行训 3 个 LoRA
#   bash run_autodl_lora.sh medical      # 只训 lora_medical
#   bash run_autodl_lora.sh all wandb    # + swanlab
#
# 前提：out/dpo_768.pth 已存在（本机已有，无需再跑 sft/dpo）
# ============================================================
set -e

MODE="${1:-all}"
WANDB_FLAG=""
[ "${2:-}" = "wandb" ] && WANDB_FLAG="--use_wandb"
[ "${1:-}" = "wandb" ] && { WANDB_FLAG="--use_wandb"; MODE="all"; }

# 脚本所在目录即项目根（不 clone、不装依赖）
PROJ="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ"

BATCH_SIZE=64
MAX_SEQ_LEN=340
EPOCHS=10
NUM_WORKERS=8
FROM_WEIGHT="dpo"

LORA_DATASETS=(lora_identity.jsonl lora_medical.jsonl lora_exam.jsonl)
LORA_NAMES=(lora_identity lora_medical lora_exam)

pick_loras() {
  case "$MODE" in
    all)      echo "0 1 2" ;;
    identity) echo "0" ;;
    medical)  echo "1" ;;
    exam)     echo "2" ;;
    *)
      echo "用法：bash run_autodl_lora.sh [all|identity|medical|exam] [wandb]"
      exit 1 ;;
  esac
}

echo ">>> LoRA | 基座：$FROM_WEIGHT | 项目：$PROJ"

# 权重检查
if [ ! -f "out/dpo_768.pth" ]; then
  echo "！！ 缺少 out/dpo_768.pth"; exit 1
fi
echo ">>> [1/3] 已找到 out/dpo_768.pth"

# 下载 LoRA 数据集（缺哪个下哪个）
echo ">>> [2/3] 准备 LoRA 数据集"
source /etc/network_turbo 2>/dev/null || true
INDICES=($(pick_loras))
FILES=("${LORA_DATASETS[@]}")
[ "$MODE" != "all" ] && FILES=() && for i in "${INDICES[@]}"; do FILES+=("${LORA_DATASETS[$i]}"); done
mkdir -p dataset
for f in "${FILES[@]}"; do
  if [ -f "dataset/$f" ]; then
    echo "    已有 dataset/$f"
  else
    echo "    下载 dataset/$f ..."
    pip install -q modelscope 2>/dev/null || true
    modelscope download --dataset gongjy/minimind_dataset "$f" --local_dir ./dataset
  fi
done

# 串行训练
echo ">>> [3/3] 串行训练 LoRA（单卡 cuda:0）"
mkdir -p out
cd trainer
for i in "${INDICES[@]}"; do
  name="${LORA_NAMES[$i]}"
  echo ""
  echo "---- [$name] 开始 ----"
  python -u train_lora.py \
    --device cuda:0 \
    --dtype bfloat16 \
    --lora_name "$name" \
    --data_path "../dataset/${LORA_DATASETS[$i]}" \
    --from_weight "$FROM_WEIGHT" \
    --batch_size "$BATCH_SIZE" \
    --max_seq_len "$MAX_SEQ_LEN" \
    --num_workers "$NUM_WORKERS" \
    --epochs "$EPOCHS" \
    --save_interval 500 \
    --log_interval 10 \
    $WANDB_FLAG \
    2>&1 | tee "../out/${name}.log"
  echo "---- [$name] 完成 -> out/${name}_768.pth ----"
done

echo ""
echo "完成。测试：python eval_llm.py --weight dpo --lora_weight lora_medical"
