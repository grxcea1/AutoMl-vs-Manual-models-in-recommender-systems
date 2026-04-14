import json  # used to read saved results from files
import matplotlib  # library for plotting graphs
matplotlib.use("agg")  # 'agg' allows saving graphs without opening a window
import matplotlib.pyplot as plt  # plt is used to create charts


# load results from manual models file
with open('results_manual.json', 'r') as f:
    manual_results = json.load(f)  # converts JSON into Python dictionary

# load results from AutoML models file
with open('results_automl.json', 'r') as f:
    automl_results = json.load(f)

all_results = {**manual_results, **automl_results} # combine both dictionaries into one (manual + automl)

model_names = list(all_results.keys()) # get all model names (used for x-axis)

manual_names = list(manual_results.keys()) # get only manual model names (used for colouring)


# sets colours for auto and manual models
colors = []
for name in model_names:
    if name in manual_names:
        colors.append("#24BFF3")  # blue = manual models
    else:
        colors.append("#FE2D88")  # pink = AutoML models


# list of metrics to plot
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

for key, ylabel, title, lower_is_better in metrics: # loops through each metric and creates a chart

    vals = [all_results[m][key] for m in model_names]  # gets values for this metric for each model
    plt.figure(figsize=(12, 5))  # wide chart so all model names fit clearly

   
    # creates a bar chart
    bars = plt.bar(model_names, vals,
                   color=colors, edgecolor='white')

    # sets title and axis labels
    plt.title(title, fontsize=13, fontweight='bold')
    plt.ylabel(ylabel, fontsize=11)

    # increases the y-axis height so labels above bars don't get cut off
    plt.ylim(0, max(vals) * 1.3)

    plt.xlabel(
        'Blue = Manual Models    Orange = AutoML    '
        ,fontsize=9, color='gray')

    # add the exact value on top of each bar
    for bar, v in zip(bars, vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # centre of bar
            bar.get_height() + 0.005,           # slightly above bar
            f'{v:.4f}',                         # format to 4 decimal places
            ha='center', va='bottom', fontsize=9)

    plt.xticks(rotation=15) #rotates x labels so they don’t overlap

    plt.tight_layout()  # adjust spacing so nothing is cut off


    # save chart as PNG file
    fname = f"chart_{key.lower()}_combined.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')

    plt.close()  # close figure to free memory

    print(f"Saved: {fname}")  # confirmation message


# final message when all charts are done
print("\nAll 6 combined charts saved")