"""从训练日志画 loss / lr 曲线。
用法：python eval/plot_loss.py --log out/pretrain.log --out out/pretrain_loss.png
"""
import os
import re
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument('--log', default=os.path.join(PROJECT_ROOT, 'out', 'pretrain.log'))
parser.add_argument('--out', default=os.path.join(PROJECT_ROOT, 'out', 'pretrain_loss.png'))
parser.add_argument('--title', default='Pretrain Loss Curve')
args = parser.parse_args()

# 解析日志：Epoch:[1/1](100/19848), loss: 7.3283, ..., lr: 0.00049997, ...
pat = re.compile(r'\((\d+)/(\d+)\),\s*loss:\s*([\d.]+).*?lr:\s*([\d.eE+-]+)')
steps, losses, lrs = [], [], []
total = None
with open(args.log, encoding='utf-8') as f:
    for line in f:
        m = pat.search(line)
        if m:
            cur, total, loss, lr = m.groups()
            steps.append(int(cur))
            losses.append(float(loss))
            lrs.append(float(lr))

if not steps:
    raise SystemExit("没解析到 loss 数据，检查日志格式")

print(f"解析到 {len(steps)} 个数据点 | 起始 loss={losses[0]:.3f} | 最终 loss={losses[-1]:.3f}")

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(steps, losses, color='tab:blue', linewidth=1.3, label='loss')
ax1.set_xlabel(f'step (total {total})')
ax1.set_ylabel('loss', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True, alpha=0.3)
# 标注起点和终点
ax1.annotate(f'{losses[0]:.2f}', (steps[0], losses[0]), color='tab:blue')
ax1.annotate(f'{losses[-1]:.2f}', (steps[-1], losses[-1]), color='tab:blue')

# 第二个 y 轴画学习率
ax2 = ax1.twinx()
ax2.plot(steps, lrs, color='tab:orange', linewidth=1.0, alpha=0.7, label='lr')
ax2.set_ylabel('learning rate', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

plt.title(args.title)
fig.tight_layout()
plt.savefig(args.out, dpi=130)
print(f">>> 已保存图像 {args.out}")
