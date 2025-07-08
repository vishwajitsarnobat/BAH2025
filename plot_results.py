import matplotlib.pyplot as plt
import sys
import pandas as pd
import re
import os
import json

def parse_log_file(filepath):
    """Parses a log file to extract the final epoch's IoU scores."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Warning: Log file not found at {filepath}. Skipping.")
        return None, None, None

    # This regex is robust for finding the last epoch's content
    epochs = re.findall(r"--- Epoch \d+/\d+ ---(.*?)(?=(--- Epoch|\Z))", content, re.S)
    if not epochs:
        print(f"Warning: No complete epochs found in {filepath}. Skipping.")
        return None, None, None
        
    last_epoch_content = epochs[-1][0]
    
    miou_match = re.search(r"Mean IoU \(mIoU\): ([\d.]+)", last_epoch_content)
    miou = float(miou_match.group(1)) if miou_match else 0.0

    class_names, iou_scores = [], []
    iou_matches = re.findall(r"Class '(.*?)' IoU: ([\d.]+)", last_epoch_content)
    for name, score in iou_matches:
        class_names.append(name)
        iou_scores.append(float(score))
        
    if not iou_scores: 
        return None, None, None
    return class_names, iou_scores, miou

def create_class_specific_plots(results):
    """Generates and saves one plot per class, ranking model performance."""
    if not results:
        print("No results to plot.")
        return

    # Create a single, master DataFrame from all results
    master_df_list = []
    class_names = next(iter(results.values()))['class_names']
    for exp_name_pretty, data in results.items():
        temp_df = pd.DataFrame({
            'IoU': data['iou_scores'],
            'Experiment': exp_name_pretty,
            'Class': data['class_names']
        })
        master_df_list.append(temp_df)
    master_df = pd.concat(master_df_list, ignore_index=True)

    output_dir = "class_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving class-specific plots to '{output_dir}/'")

    # Use a visually appealing colormap
    color_map = plt.cm.get_cmap('plasma', len(results))

    for i, class_name in enumerate(class_names):
        class_df = master_df[master_df['Class'] == class_name].sort_values(by='IoU', ascending=False).reset_index(drop=True)
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(class_df) * 0.8))) # Dynamic height
        
        bar_colors = [color_map(j / len(class_df)) for j in range(len(class_df))]
        bars = ax.barh(class_df['Experiment'], class_df['IoU'], color=bar_colors)
        
        ax.set_xlabel('Intersection-over-Union (IoU)', fontsize=12)
        ax.set_title(f"Model Performance for Class: '{class_name.upper()}'", fontsize=16, weight='bold')
        ax.set_xlim(0, 1.0)
        ax.invert_yaxis()
        ax.grid(axis='x', linestyle='--', alpha=0.6)

        for index, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2., f' {width:.3f}', va='center', fontsize=11)
            ax.text(-0.01, bar.get_y() + bar.get_height()/2., f'#{index+1}', va='center', ha='right', fontsize=14, weight='bold', color='dimgray')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', length=0, labelsize=12, pad=-5, labeltop=False, labelbottom=True)
        plt.setp(ax.get_yticklabels(), ha="left")

        fig.tight_layout(pad=2)
        plt.savefig(os.path.join(output_dir, f"{i:02d}_{class_name.replace(' ', '_')}.png"), dpi=150)
        plt.close(fig)

if __name__ == "__main__":
    try:
        with open('experiments.json', 'r') as f:
            experiments = json.load(f)['experiments']
    except FileNotFoundError:
        print("Error: experiments.json not found. Please create it.")
        sys.exit(1)
    
    experiment_map = {exp['name_code']: exp['name_pretty'] for exp in experiments}
    all_results = {}

    for name_code, name_pretty in experiment_map.items():
        log_path = os.path.join("logs", f"{name_code}_logs.txt")
        if os.path.exists(log_path):
            class_names, iou_scores, miou = parse_log_file(log_path)
            if class_names:
                all_results[name_pretty] = {
                    'class_names': class_names,
                    'iou_scores': iou_scores,
                    'miou': miou
                }
    
    if all_results:
        create_class_specific_plots(all_results)
    else:
        print("No valid log files found to plot. Please run the experiments first.")