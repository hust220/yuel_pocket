
import pandas as pd
from pathlib import Path

def compare_success_rates():
    base_dir = Path("/home/tyq4zn/scratch/codes/yuel_pocket/benchmark")
    
    # Input Files
    gt_csv_path = base_dir / "plinder/evaluate_residues2_test1036_residues.csv"
    af_csv_path = base_dir / "alphafold/rmsd_success_test1036_af.csv"
    output_csv_path = base_dir / "alphafold/comparison_af_vs_gt_success.csv"
    
    # Load DataFrames
    print(f"Loading Ground Truth (Plinder) results from {gt_csv_path}...")
    if not gt_csv_path.exists():
        print(f"Error: {gt_csv_path} does not exist.")
        return

    print(f"Loading AlphaFold results from {af_csv_path}...")
    if not af_csv_path.exists():
        print(f"Error: {af_csv_path} does not exist.")
        return

    df_gt = pd.read_csv(gt_csv_path)
    df_af = pd.read_csv(af_csv_path)
    
    # Check required columns
    if 'Success_Rank3_Dist4' not in df_gt.columns:
        print("Error: 'Success_Rank3_Dist4' column missing in Ground Truth CSV.")
        return
    if 'Success' not in df_af.columns:
        print("Error: 'Success' column missing in AlphaFold CSV.")
        return
        
    # Select relevant columns and rename for clarity
    df_gt_subset = df_gt[['SystemID', 'Success_Rank3_Dist4']].rename(columns={'Success_Rank3_Dist4': 'GT_Success'})
    df_af_subset = df_af[['SystemID', 'Success', 'RMSD']].rename(columns={'Success': 'AF_Success'})
    
    # Merge DataFrames
    # Use inner join to compare only systems present in both (which should be most/all)
    merged_df = pd.merge(df_gt_subset, df_af_subset, on='SystemID', how='inner')
    
    print(f"Merged {len(merged_df)} common systems.")
    
    # Define Categories
    def categorize(row):
        gt = row['GT_Success']
        af = row['AF_Success']
        if gt == 1 and af == 1:
            return "Both_Success"
        elif gt == 0 and af == 0:
            return "Both_Fail"
        elif gt == 1 and af == 0:
            return "GT_Pass_AF_Fail"
        elif gt == 0 and af == 1:
            return "GT_Fail_AF_Pass"
        else:
            return "Unknown"

    merged_df['Category'] = merged_df.apply(categorize, axis=1)
    
    # Save Results
    merged_df.to_csv(output_csv_path, index=False)
    print(f"Comparison results saved to {output_csv_path}")
    
    # Print Summary
    summary = merged_df['Category'].value_counts()
    print("\nSummary Counts:")
    print(summary)
    
    # Print examples for "Interesting" cases
    print("\n--- Examples: GT Failed but AF Passed (False Positive / Robustness?) ---")
    print(merged_df[merged_df['Category'] == 'GT_Fail_AF_Pass'][['SystemID', 'RMSD']].head().to_string(index=False))
    
    print("\n--- Examples: GT Passed but AF Failed (Sensitivity Loss due to folding?) ---")
    print(merged_df[merged_df['Category'] == 'GT_Pass_AF_Fail'][['SystemID', 'RMSD']].head().to_string(index=False))

if __name__ == "__main__":
    compare_success_rates()
