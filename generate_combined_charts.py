import json
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


with open('results_manual.json', 'r') as f:
    manual_results = json.load(f)

with open('results_automl.json', 'r') as f:
    automl_results = json.load(f)

all_results = {**manual_results, **automl_results}

model_names = list(all_results.keys())

manual_names = list(manual_results.keys())
colors = []
for name in model_names:
    if name in manual_names:
        colors.append("#24BFF3")  # blue = manual
    else:
        colors.append("#FE2D88")  # orange = automl

metrics = [
    ('Precision@k', 'Precision@10',
     'Which model gave the most relevant top 10?', False),
    ('Recall@k', 'Recall@10',
     'Which model found the most liked movies in top 10?', False),
    ('Precision', 'Precision (full test set)',
     'Which model had the highest overall precision?', False),
    ('Recall', 'Recall (full test set)',
     'Which model had the highest overall recall?', False),
    ('RMSE', 'RMSE',
     'Which model predicted ratings most accurately?', True),
    ('Runtime', 'Runtime (seconds)',
     'Which model was fastest?', True),
]


for key, ylabel, title, lower_is_better in metrics:

    vals = [all_results[m][key] for m in model_names]
    # gets the value for this metric for each model

    plt.figure(figsize=(12, 5))
    # wider figure so 6 model names fit without overlapping

    bars = plt.bar(model_names, vals,
                   color=colors, edgecolor='white')

    plt.title(title, fontsize=13, fontweight='bold')
    plt.ylabel(ylabel, fontsize=11)
    plt.ylim(0, max(vals) * 1.3)
    # 1.3 = 30% higher than tallest bar so numbers on top dont get clipped

    note = 'lower is better' if lower_is_better \
        else 'higher is better'
    plt.xlabel(
        'Blue = Manual Models    Orange = AutoML    '
        + note, fontsize=9, color='gray')

    # writes value on top of each bar so reader can see exact number
    for bar, v in zip(bars, vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{v:.4f}',
            ha='center', va='bottom', fontsize=9)

    plt.xticks(rotation=15)
    # rotates model names 15 degrees so they dont overlap
    plt.tight_layout()

    fname = f"chart_{key.lower()}_combined.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fname}")

print("\nAll 6 combined charts saved")



