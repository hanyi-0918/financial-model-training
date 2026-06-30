"""通用评测脚本：用固定问题集评测任意阶段的权重，结果写入带阶段标签的文件。

用法示例：
  python eval/run_eval.py --weight pretrain                 # 预训练
  python eval/run_eval.py --weight full_sft                 # SFT
  python eval/run_eval.py --weight rlhf                      # DPO
  python eval/run_eval.py --weight full_sft --lora_weight lora_finance  # LoRA

脚本不变，只换 --weight，即可对同一批问题做横向对比。
"""
import os
import sys
import time
import argparse
import random
import warnings
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora
from trainer.trainer_utils import setup_seed
warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser(description="MiniMind 阶段对比评测")
    parser.add_argument('--weight', default='pretrain', type=str, help="权重前缀(pretrain/full_sft/rlhf/...)")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名(None=不用)")
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--max_new_tokens', default=200, type=int)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.95, type=float)
    parser.add_argument('--prompts', default=os.path.join(PROJECT_ROOT, 'eval', 'eval_prompts.txt'), type=str)
    parser.add_argument('--out_dir', default=os.path.join(PROJECT_ROOT, 'eval'), type=str)
    parser.add_argument('--save_dir', default=os.path.join(PROJECT_ROOT, 'out'), type=str)
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'), type=str)
    args = parser.parse_args()

    # 读问题集（跳过注释和空行）
    with open(args.prompts, encoding='utf-8') as f:
        prompts = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')]

    # 加载 tokenizer + 模型权重
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(PROJECT_ROOT, 'model'))
    model = MiniMindForCausalLM(MiniMindConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)))
    moe_suffix = '_moe' if args.use_moe else ''
    ckp = os.path.join(args.save_dir, f'{args.weight}_{args.hidden_size}{moe_suffix}.pth')
    model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    if args.lora_weight != 'None':
        apply_lora(model)
        load_lora(model, os.path.join(args.save_dir, f'{args.lora_weight}_{args.hidden_size}.pth'))
    model = model.half().eval().to(args.device)

    tag = args.weight + ('' if args.lora_weight == 'None' else f'+{args.lora_weight}')
    results = [f"========== 评测：{tag} ==========",
               f"权重文件：{os.path.basename(ckp)} | 设备：{args.device} | max_new_tokens={args.max_new_tokens}\n"]

    for p in prompts:
        setup_seed(42)  # 固定种子，保证可复现对比
        if 'pretrain' in args.weight:
            inputs_text = tokenizer.bos_token + p          # 预训练：续写模式
        else:
            inputs_text = tokenizer.apply_chat_template(    # 对齐后：对话模式
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(args.device)
        st = time.time()
        out = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, top_p=args.top_p,
            temperature=args.temperature, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, repetition_penalty=1)
        resp = tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        speed = (len(out[0]) - len(inputs["input_ids"][0])) / (time.time() - st)
        block = f"💬: {p}\n🧠: {resp}\n[Speed]: {speed:.1f} tokens/s\n"
        results.append(block)
        print(block)

    out_path = os.path.join(args.out_dir, f'eval_{tag.replace("+", "_")}.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(results))
    print(f">>> 已写入 {out_path}")


if __name__ == "__main__":
    main()
