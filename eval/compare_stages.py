"""三阶段对比评测：对同一批问题，跑 pretrain / full_sft / dpo 三个权重，
并排输出回答 + 自动行为指标，直观看出"续写→对话→对齐"的进化。

用法：
  python eval/compare_stages.py
  python eval/compare_stages.py --stages pretrain full_sft dpo --max_new_tokens 200

产出：eval/compare_3stages.txt（并排回答 + 每条指标 + 末尾汇总表）
"""
import os
import sys
import argparse
import warnings
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed
warnings.filterwarnings('ignore')


def distinct_ngram_ratio(token_ids, n=3):
    """distinct-n：唯一 n-gram 占比，越低=重复越多。"""
    if len(token_ids) < n:
        return 1.0
    grams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    return len(set(grams)) / len(grams)


def load_model(weight, hidden_size, num_layers, save_dir, device):
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_layers))
    ckp = os.path.join(save_dir, f'{weight}_{hidden_size}.pth')
    model.load_state_dict(torch.load(ckp, map_location=device), strict=True)
    return model.half().eval().to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', nargs='+', default=['pretrain', 'full_sft', 'dpo'])
    ap.add_argument('--hidden_size', type=int, default=768)
    ap.add_argument('--num_hidden_layers', type=int, default=8)
    ap.add_argument('--max_new_tokens', type=int, default=512)
    ap.add_argument('--temperature', type=float, default=0.85)
    ap.add_argument('--top_p', type=float, default=0.95)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--prompts', default=os.path.join(PROJECT_ROOT, 'eval', 'eval_prompts.txt'))
    ap.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'eval', 'compare_3stages.txt'))
    ap.add_argument('--save_dir', default=os.path.join(PROJECT_ROOT, 'out'))
    ap.add_argument('--device', default='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    args = ap.parse_args()

    with open(args.prompts, encoding='utf-8') as f:
        prompts = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith('#')]

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(PROJECT_ROOT, 'model'))

    # results[stage][i] = {text, n_tokens, stopped, distinct3}
    results = {}
    agg = {}  # 聚合指标
    for stage in args.stages:
        print(f'>>> 评测 {stage} ...')
        model = load_model(stage, args.hidden_size, args.num_hidden_layers, args.save_dir, args.device)
        results[stage] = []
        tot_len, n_stopped, tot_distinct, min_distinct = 0, 0, 0.0, 1.0
        for p in prompts:
            setup_seed(args.seed)  # 同种子，公平对比
            if 'pretrain' in stage:
                text_in = tokenizer.bos_token + p                       # 续写模式
            else:
                text_in = tokenizer.apply_chat_template(                # 对话模式
                    [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text_in, return_tensors="pt", truncation=True).to(args.device)
            out = model.generate(
                inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens, do_sample=True, top_p=args.top_p,
                temperature=args.temperature, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)
            gen_ids = out[0][len(inputs["input_ids"][0]):].tolist()
            # 去掉尾部 pad
            gen_ids = [t for t in gen_ids if t != tokenizer.pad_token_id]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            n_tok = len(gen_ids)
            stopped = n_tok < args.max_new_tokens          # 没顶满=自己输出了eos=会停
            d3 = distinct_ngram_ratio(gen_ids, 3)
            results[stage].append({'text': text, 'n': n_tok, 'stopped': stopped, 'd3': d3})
            tot_len += n_tok; n_stopped += int(stopped); tot_distinct += d3
            min_distinct = min(min_distinct, d3)
        np_ = len(prompts)
        agg[stage] = {'avg_len': tot_len / np_, 'stop_rate': n_stopped / np_,
                      'avg_distinct3': tot_distinct / np_, 'min_distinct3': min_distinct}
        del model

    # ===== 写报告 =====
    lines = ["=" * 70, "三阶段对比评测：pretrain → full_sft → dpo",
             f"问题数 {len(prompts)} | max_new_tokens={args.max_new_tokens} | 固定种子={args.seed}", "=" * 70, ""]

    for i, p in enumerate(prompts):
        lines.append(f"【问题 {i+1}】{p}")
        for stage in args.stages:
            r = results[stage][i]
            flag = "✓停" if r['stopped'] else "✗顶满"
            lines.append(f"  ── [{stage}] (长度{r['n']} {flag} 非重复率{r['d3']:.2f})")
            lines.append(f"     {r['text']}")
        lines.append("")

    # 汇总表
    lines += ["=" * 70, "聚合指标汇总（看整体进化）", "-" * 70,
              f"{'阶段':<12}{'平均长度':<10}{'会停比例':<10}{'平均非重复率':<14}{'最低非重复率(最差案例)':<20}"]
    for stage in args.stages:
        a = agg[stage]
        lines.append(f"{stage:<12}{a['avg_len']:<10.1f}{a['stop_rate']*100:<9.0f}%{a['avg_distinct3']:<14.2f}{a['min_distinct3']:<20.2f}")
    lines += ["-" * 70,
              "解读：会停比例高=学会输出eos；非重复率越高越好；",
              "     最低非重复率暴露最差案例（pretrain 常有废话循环，该值很低）。", "=" * 70]

    report = "\n".join(lines)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(report)
    print("\n" + report)
    print(f"\n>>> 已写入 {args.out}")


if __name__ == "__main__":
    main()
