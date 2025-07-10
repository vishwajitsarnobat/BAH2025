"""
Parses segmentation model training logs, extracts final epoch IoU scores,
and generates per-class bar plots comparing the performance of different models.

This version is corrected to fix the critical data-handling bug that caused
incorrect plotting results, along with other minor improvements.
"""
import matplotlib.pyplot as plt
import sys
import pandas as pd
import re
import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple

def parse_log_file(filepath: Path) -> Optional[Tuple[List[str], List[float], float]]:
    """
    Parses a log file to extract the final epoch's class names and IoU scores.
    """
    try:
        content = filepath.read_text()
    except FileNotFoundError:
        return None

    # This regex is robust. It looks for the LAST occurrence of "--- Validation Metrics ---"
    # and captures everything after it.
    matches = re.findall(r"--- Validation Metrics ---(.*?)---", content, re.S)
    if not matches:
        # Fallback for files that might not have a trailing '---'
        matches = re.findall(r"--- Validation Metrics ---(.*)", content, re.S)
    
    if not matches:
        print(f"    -> ⚠️ DEBUG: Could not find a '--- Validation Metrics ---' block in {filepath.name}.")
        return None
        
    final_metrics_content = matches[-1]
    
    miou_match = re.search(r"Mean IoU \(mIoU\):\s*([\d.]+)", final_metrics_content)
    miou = float(miou_match.group(1)) if miou_match else 0.0

    iou_matches = re.findall(r"^\s*Class\s*'(.*?)'\s*IoU:\s*([\d.]+)", final_metrics_content, re.MULTILINE)
    
    if not iou_matches:
        print(f"    -> ⚠️ DEBUG: Found metrics block, but failed to match `Class '...'` lines in {filepath.name}.")
        return None

    class_names, iou_scores = [], []
    for name, score_str in iou_matches:
        class_names.append(name)
        iou_scores.append(float(score_str))
        
    return class_names, iou_scores, miou

def create_class_specific_plots(results: Dict[str, Dict], output_dir: Path):
    """
    Generates and saves one plot per class, ranking model performance based on IoU.
    """
    if not results:
        print("No valid results found to plot.")
        return

    # --- CORRECTED DATAFRAME CREATION ---
    # Build a list of dictionaries, then create the DataFrame once. This is robust.
    all_rows = []
    for exp_name_pretty, data in results.items():
        for class_name, iou_score in zip(data['class_names'], data['iou_scores']):
            all_rows.append({
                'Experiment': exp_name_pretty,
                'Class': class_name,
                'IoU': iou_score
            })
    master_df = pd.DataFrame(all_rows)

    if master_df.empty:
        print("Master DataFrame is empty. Cannot create plots.")
        return

    output_dir.mkdir(exist_ok=True)
    class_names = master_df['Class'].unique()
    print(f"\nSaving {len(class_names)} class-specific plots to '{output_dir}/'")

    color_map = plt.get_cmap('plasma')
    plt.style.use('seaborn-v0_8-whitegrid')

    for i, class_name in enumerate(class_names):
        class_df = master_df[master_df['Class'] == class_name].sort_values(
            by='IoU', ascending=False
        ).reset_index(drop=True)
        
        num_bars = len(class_df)
        if num_bars == 0: continue
            
        fig, ax = plt.subplots(figsize=(12, max(6, num_bars * 0.7)))
        
        bar_colors = [color_map(j / (num_bars - 1)) if num_bars > 1 else color_map(0.5) for j in range(num_bars)]
        bars = ax.barh(class_df['Experiment'], class_df['IoU'], color=bar_colors, height=0.7)
        
        ax.set_xlabel('Intersection-over-Union (IoU)', fontsize=12)
        ax.set_title(f"Model Performance for Class: '{class_name.upper()}'", fontsize=16, weight='bold', pad=20)
        ax.set_xlim(0, 1.05)
        ax.invert_yaxis()
        
        ax.grid(visible=True, which='major', axis='x', linestyle='--', alpha=0.6)
        ax.grid(visible=False, which='major', axis='y')

        for index, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.005, bar.get_y() + bar.get_height()/2., f'{width:.3f}', va='center', fontsize=11)
            ax.text(-0.01, bar.get_y() + bar.get_height()/2., f'#{index+1}', va='center', ha='right', fontsize=14, weight='bold', color='dimgray')
        
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0, labelsize=12, pad=5)
        plt.setp(ax.get_yticklabels(), ha="right")

        fig.tight_layout(pad=2)
        safe_class_name = class_name.replace(' ', '_').replace('/', '_')
        plot_path = output_dir / f"{i:02d}_{safe_class_name}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)

def main():
    """Main execution function."""
    config_file = Path('experiments.json')
    log_dir = Path("logs")
    output_dir = Path("class_plots")

    if not config_file.is_file():
        print(f"Error: Configuration file not found at '{config_file}'. Please create it.")
        sys.exit(1)

    with open(config_file, 'r') as f:
        experiments = json.load(f)['experiments']
    
    experiment_map = {exp['name_code']: exp['name_pretty'] for exp in experiments}
    print(f"✅ Found {len(experiment_map)} experiments defined in '{config_file}'.")
    print("-" * 40)

    all_results = {}
    for name_code, name_pretty in experiment_map.items():
        print(f"Processing model: '{name_pretty}' ({name_code})")
        log_path = log_dir / f"{name_code}_logs.txt"
        
        if log_path.is_file():
            print(f"  -> Found log file: '{log_path}'")
            parsed_data = parse_log_file(log_path)
            if parsed_data:
                class_names, iou_scores, miou = parsed_data
                print(f"  -> ✅ Successfully parsed data. mIoU: {miou:.4f}")
                all_results[name_pretty] = { 'class_names': class_names, 'iou_scores': iou_scores, 'miou': miou }
            else:
                print(f"  -> ❌ ERROR: Log file exists but could not be parsed. Check its content.")
        else:
            print(f"  -> ❌ ERROR: Log file not found. Skipping this model.")
        print("-" * 20)

    print("-" * 40)
    print(f"Found valid results for {len(all_results)} out of {len(experiment_map)} models.")
    
    if all_results:
        create_class_specific_plots(all_results, output_dir)
        print("\n✨ Plot generation complete.")
    else:
        print("\nNo valid log files could be processed. Please check your 'logs' directory and 'experiments.json' file.")

if __name__ == "__main__":
    main()