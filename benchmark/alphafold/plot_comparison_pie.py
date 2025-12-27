
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_comparison_pie_chart():
    csv_path = "/home/tyq4zn/scratch/codes/yuel_pocket/benchmark/alphafold/comparison_af_vs_gt_success.csv"
    output_png = "/home/tyq4zn/scratch/codes/yuel_pocket/benchmark/alphafold/af_vs_gt_comparison_pie.png"
    output_svg = "/home/tyq4zn/scratch/codes/yuel_pocket/benchmark/alphafold/af_vs_gt_comparison_pie.svg"
    
    if not Path(csv_path).exists():
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    counts = df['Category'].value_counts()
    
    # Mapping to nicer labels and colors
    # Category names from previous script: Both_Success, Both_Fail, GT_Pass_AF_Fail, GT_Fail_AF_Pass
    
    label_map = {
        'Both_Success': 'Both Success',
        'Both_Fail': 'Both Fail',
        'GT_Pass_AF_Fail': 'GT Pass / AF Fail',
        'GT_Fail_AF_Pass': 'GT Fail / AF Pass'
    }
    
    # Define colors
    # Success: Blueish (#43A3EF)
    # Fail: Reddish (#EF767B)
    # Mixed: Maybe Purple/Orange or variations
    
    # Let's use a distinct palette
    colors_map = {
        'Both_Success': '#43A3EF',      # Blue (Good)
        'Both_Fail': '#EF767B',          # Red (Bad)
        'GT_Pass_AF_Fail': '#F5A623',    # Orange (Regression)
        'GT_Fail_AF_Pass': '#7ED321'     # Green (Improvement/Surprise)
    }
    
    labels = [label_map.get(idx, idx) for idx in counts.index]
    colors = [colors_map.get(idx, 'gray') for idx in counts.index]
    
    # Plot
    plt.figure(figsize=(2.7, 2.2))
    
    # Pie chart
    wedges, texts, autotexts = plt.pie(counts, labels=labels, autopct='%1.1f%%', 
                                       colors=colors, startangle=140, 
                                       wedgeprops=dict(width=0.5), # Donut style looks nice
                                       textprops=dict(fontsize=8))
    
    # Styling
    plt.setp(texts, fontweight=500)
    plt.setp(autotexts, size=8, weight="bold", color="white")
    
    # Ensure circle
    plt.axis('equal')  
    
    # Title? Maybe skip or keep simple
    # plt.title('AlphaFold vs Ground Truth Prediction Comparison', fontsize=10)
    
    plt.tight_layout()
    
    plt.savefig(output_png, dpi=300)
    plt.savefig(output_svg, dpi=300)
    plt.close()
    
    print(f"Pie chart saved to {output_png} and {output_svg}")
    print("Counts:")
    print(counts)

if __name__ == "__main__":
    plot_comparison_pie_chart()
