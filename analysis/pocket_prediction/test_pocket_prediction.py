#%%

import os
import torch
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('../../')
from src.lightning import YuelPocket
from src.db_utils import db_connection, add_column
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def evaluate_predictions(predictions, targets, masks, all_pockets=None, threshold=0.06):
    """Evaluate predictions using success rate based on overlap between predicted and true pockets.
    Three success metrics are calculated:
    1. Original: True if there is any overlap between predicted pockets (prob > threshold) and true pockets
    2. Max-based: True if the residue with maximum probability is a true pocket residue
    3. All-pockets: True if the residue with maximum probability is in the combined pockets from all ligands
    """
    # Get valid indices where mask == 1
    valid_indices = np.where(masks == 1)[0]
    valid_preds = predictions[valid_indices]
    valid_targets = targets[valid_indices]
    
    # Check if we have valid data
    if len(valid_targets) == 0:
        print("Warning: No valid targets found after masking")
        return None
    
    # Check if targets contain both classes
    unique_targets = np.unique(valid_targets)
    if len(unique_targets) == 1:
        if unique_targets[0] == 0:
            print("Warning: All targets are 0, cannot calculate metrics")
            return None
        elif unique_targets[0] == 1:
            print("Warning: All targets are 1, cannot calculate metrics")
            return None
    
    try:
        # Calculate ROC and PR metrics
        fpr, tpr, _ = roc_curve(valid_targets, valid_preds)
        roc_auc = auc(fpr, tpr)
        ap = average_precision_score(valid_targets, valid_preds)
        
        # Calculate IQR-based metrics
        q1 = np.percentile(valid_preds, 25)
        q3 = np.percentile(valid_preds, 75)
        iqr = q3 - q1
        # Calculate threshold using Q3 + 1.5*IQR
        iqr_threshold = q3 + 1.5 * iqr
        # Create binary predictions using IQR threshold
        valid_predictions_iqr = valid_preds > iqr_threshold
        # Calculate ROC and PR metrics using binary predictions
        fpr_iqr, tpr_iqr, _ = roc_curve(valid_targets, valid_predictions_iqr.astype(float))
        roc_auc_iqr = auc(fpr_iqr, tpr_iqr)
        ap_iqr = average_precision_score(valid_targets, valid_predictions_iqr.astype(float))
        
        # Find predicted pocket positions (prob > threshold)
        pred_pockets = (valid_preds > threshold)
        true_pockets = (valid_targets == 1)
        
        # Calculate positive prediction ratio and true pocket ratio
        positive_ratio = np.mean(pred_pockets)
        true_ratio = np.mean(true_pockets)
        iqr_ratio = np.mean(valid_preds > iqr_threshold)
        
        # Get non-zero predictions
        nonzero_preds = valid_preds[valid_preds > 0]
        
        # Calculate ratios for different thresholds
        thresholds = np.linspace(0, 1, 21)  # 0 to 1 in 21 steps (including both ends)
        threshold_ratios = []
        for threshold in thresholds:
            ratio = np.mean(valid_preds > threshold)
            threshold_ratios.append(ratio)
        
        # Calculate overlap
        overlap = pred_pockets & true_pockets
        
        # Calculate confusion matrix metrics
        TP = np.sum(overlap)  # True Positives
        FP = np.sum(pred_pockets & ~true_pockets)  # False Positives
        TN = np.sum(~pred_pockets & ~true_pockets)  # True Negatives
        FN = np.sum(~pred_pockets & true_pockets)  # False Negatives
        
        # Calculate additional metrics
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN)
                
        # Calculate Matthews Correlation Coefficient
        numerator = (TP * TN) - (FP * FN)
        denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        mcc = numerator / denominator if denominator > 0 else 0
        
        # Calculate F1 and F2 scores
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate Positive Predictive Value (PPV) and Negative Predictive Value (NPV)
        npv = TN / (TN + FN) if (TN + FN) > 0 else 0

        # Calculate IQR-based confusion matrix metrics
        TP_iqr = np.sum(valid_predictions_iqr & true_pockets)  # True Positives
        FP_iqr = np.sum(valid_predictions_iqr & ~true_pockets)  # False Positives
        TN_iqr = np.sum(~valid_predictions_iqr & ~true_pockets)  # True Negatives
        FN_iqr = np.sum(~valid_predictions_iqr & true_pockets)  # False Negatives
        
        # Calculate IQR-based additional metrics
        precision_iqr = TP_iqr / (TP_iqr + FP_iqr) if (TP_iqr + FP_iqr) > 0 else 0
        recall_iqr = TP_iqr / (TP_iqr + FN_iqr) if (TP_iqr + FN_iqr) > 0 else 0
        specificity_iqr = TN_iqr / (TN_iqr + FP_iqr) if (TN_iqr + FP_iqr) > 0 else 0
        accuracy_iqr = (TP_iqr + TN_iqr) / (TP_iqr + TN_iqr + FP_iqr + FN_iqr)
                
        # Calculate IQR-based Matthews Correlation Coefficient
        numerator_iqr = (TP_iqr * TN_iqr) - (FP_iqr * FN_iqr)
        denominator_iqr = np.sqrt((TP_iqr + FP_iqr) * (TP_iqr + FN_iqr) * (TN_iqr + FP_iqr) * (TN_iqr + FN_iqr))
        mcc_iqr = numerator_iqr / denominator_iqr if denominator_iqr > 0 else 0
        
        # Calculate IQR-based F1 score
        f1_score_iqr = 2 * (precision_iqr * recall_iqr) / (precision_iqr + recall_iqr) if (precision_iqr + recall_iqr) > 0 else 0
        
        # Calculate IQR-based NPV
        npv_iqr = TN_iqr / (TN_iqr + FN_iqr) if (TN_iqr + FN_iqr) > 0 else 0
        
        # Calculate metrics against all_pockets if available
        if all_pockets is not None:
            # Threshold-based metrics for all pockets
            all_pockets = all_pockets[valid_indices]
            all_pockets_overlap = pred_pockets & (all_pockets == 1)
            TP_all = np.sum(all_pockets_overlap)
            FP_all = np.sum(pred_pockets & (all_pockets == 0))
            TN_all = np.sum(~pred_pockets & (all_pockets == 0))
            FN_all = np.sum(~pred_pockets & (all_pockets == 1))
            
            precision_all = TP_all / (TP_all + FP_all) if (TP_all + FP_all) > 0 else 0
            recall_all = TP_all / (TP_all + FN_all) if (TP_all + FN_all) > 0 else 0
            specificity_all = TN_all / (TN_all + FP_all) if (TN_all + FP_all) > 0 else 0
            accuracy_all = (TP_all + TN_all) / (TP_all + TN_all + FP_all + FN_all)
            
            numerator_all = (TP_all * TN_all) - (FP_all * FN_all)
            denominator_all = np.sqrt((TP_all + FP_all) * (TP_all + FN_all) * (TN_all + FP_all) * (TN_all + FN_all))
            mcc_all = numerator_all / denominator_all if denominator_all > 0 else 0
            
            f1_score_all = 2 * (precision_all * recall_all) / (precision_all + recall_all) if (precision_all + recall_all) > 0 else 0
            npv_all = TN_all / (TN_all + FN_all) if (TN_all + FN_all) > 0 else 0

            # IQR-based metrics for all pockets
            all_pockets_overlap_iqr = valid_predictions_iqr & (all_pockets == 1)
            TP_all_iqr = np.sum(all_pockets_overlap_iqr)
            FP_all_iqr = np.sum(valid_predictions_iqr & (all_pockets == 0))
            TN_all_iqr = np.sum(~valid_predictions_iqr & (all_pockets == 0))
            FN_all_iqr = np.sum(~valid_predictions_iqr & (all_pockets == 1))
            
            precision_all_iqr = TP_all_iqr / (TP_all_iqr + FP_all_iqr) if (TP_all_iqr + FP_all_iqr) > 0 else 0
            recall_all_iqr = TP_all_iqr / (TP_all_iqr + FN_all_iqr) if (TP_all_iqr + FN_all_iqr) > 0 else 0
            specificity_all_iqr = TN_all_iqr / (TN_all_iqr + FP_all_iqr) if (TN_all_iqr + FP_all_iqr) > 0 else 0
            accuracy_all_iqr = (TP_all_iqr + TN_all_iqr) / (TP_all_iqr + TN_all_iqr + FP_all_iqr + FN_all_iqr)
            
            numerator_all_iqr = (TP_all_iqr * TN_all_iqr) - (FP_all_iqr * FN_all_iqr)
            denominator_all_iqr = np.sqrt((TP_all_iqr + FP_all_iqr) * (TP_all_iqr + FN_all_iqr) * (TN_all_iqr + FP_all_iqr) * (TN_all_iqr + FN_all_iqr))
            mcc_all_iqr = numerator_all_iqr / denominator_all_iqr if denominator_all_iqr > 0 else 0
            
            f1_score_all_iqr = 2 * (precision_all_iqr * recall_all_iqr) / (precision_all_iqr + recall_all_iqr) if (precision_all_iqr + recall_all_iqr) > 0 else 0
            npv_all_iqr = TN_all_iqr / (TN_all_iqr + FN_all_iqr) if (TN_all_iqr + FN_all_iqr) > 0 else 0
                
        # Check if there is any overlap for each sample
        has_overlap = np.any(overlap)
        
        # Calculate statistics
        n_pred_pockets = np.sum(pred_pockets)
        n_true_pockets = np.sum(true_pockets)
        n_overlap = np.sum(overlap)
        
        # Calculate max probability at true pocket positions
        true_pocket_probs = valid_preds[true_pockets]
        max_prob_at_true = np.max(true_pocket_probs) if len(true_pocket_probs) > 0 else 0
        
        # Calculate overall max probability and check if it's at a true pocket
        overall_max_prob = np.max(valid_preds)
        max_prob_idx = np.argmax(valid_preds)
        max_prob_is_pocket = valid_targets[max_prob_idx] == 1
        
        # Check if max probability position is in all_pockets
        max_prob_is_all_pocket = False
        if all_pockets is not None:
            all_pockets = all_pockets[valid_indices]  # Apply same masking as other data
            max_prob_is_all_pocket = all_pockets[max_prob_idx] == 1
        
        # Calculate rank of max probability
        all_probs_sorted = np.sort(valid_preds)[::-1]  # Sort in descending order
        rank = np.where(all_probs_sorted == max_prob_at_true)[0][0] + 1 if len(true_pocket_probs) > 0 else 0
        rank_percentile = rank / len(valid_preds) if len(valid_preds) > 0 else 0
        
        metrics = {
            'success': has_overlap,  # Original success metric: True if there is any overlap
            'max_success': max_prob_is_pocket,  # New success metric: True if max prob is at a true pocket
            'all_pockets_success': max_prob_is_all_pocket,  # Newest success metric: True if max prob is at any pocket
            'n_pred_pockets': n_pred_pockets,
            'n_true_pockets': n_true_pockets,
            'n_overlap': n_overlap,
            'n_samples': len(valid_targets),
            'max_prob_at_true': max_prob_at_true,
            'overall_max_prob': overall_max_prob,
            'rank': rank,
            'rank_percentile': rank_percentile,
            'predictions': valid_preds,  # Store predictions for later analysis
            'predictions_iqr': valid_predictions_iqr,  # Store IQR-based predictions for later analysis
            'positive_ratio': positive_ratio,  # Add positive ratio
            'true_ratio': true_ratio,  # Add true ratio
            'iqr_ratio': iqr_ratio,  # Add IQR ratio
            'nonzero_preds': nonzero_preds,  # Add non-zero predictions
            'threshold_ratios': threshold_ratios,  # Add threshold ratios
            'thresholds': thresholds,  # Add thresholds
            'is_pocket': valid_targets,  # Add is_pocket data
            'roc_auc': roc_auc,  # Add ROC AUC
            'ap': ap,  # Add Average Precision
            'roc_auc_iqr': roc_auc_iqr,  # Add IQR-based ROC AUC
            'ap_iqr': ap_iqr,  # Add IQR-based Average Precision
            'iqr_threshold': iqr_threshold,  # Add IQR threshold
            # Add confusion matrix metrics
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'accuracy': accuracy,
            'mcc': mcc,
            'f1_score': f1_score,
            'npv': npv,
            # Add IQR-based confusion matrix metrics
            'TP_iqr': TP_iqr,
            'FP_iqr': FP_iqr,
            'TN_iqr': TN_iqr,
            'FN_iqr': FN_iqr,
            'precision_iqr': precision_iqr,
            'recall_iqr': recall_iqr,
            'specificity_iqr': specificity_iqr,
            'accuracy_iqr': accuracy_iqr,
            'mcc_iqr': mcc_iqr,
            'f1_score_iqr': f1_score_iqr,
            'npv_iqr': npv_iqr,
        }

        # Add all_pockets metrics if available
        if all_pockets is not None:
            metrics.update({
                'TP_all': TP_all,
                'FP_all': FP_all,
                'TN_all': TN_all,
                'FN_all': FN_all,
                'precision_all': precision_all,
                'recall_all': recall_all,
                'specificity_all': specificity_all,
                'accuracy_all': accuracy_all,
                'mcc_all': mcc_all,
                'f1_score_all': f1_score_all,
                'npv_all': npv_all,
                'TP_all_iqr': TP_all_iqr,
                'FP_all_iqr': FP_all_iqr,
                'TN_all_iqr': TN_all_iqr,
                'FN_all_iqr': FN_all_iqr,
                'precision_all_iqr': precision_all_iqr,
                'recall_all_iqr': recall_all_iqr,
                'specificity_all_iqr': specificity_all_iqr,
                'accuracy_all_iqr': accuracy_all_iqr,
                'mcc_all_iqr': mcc_all_iqr,
                'f1_score_all_iqr': f1_score_all_iqr,
                'npv_all_iqr': npv_all_iqr,
            })
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        print(f"Predictions shape: {valid_preds.shape}, range: [{valid_preds.min():.3f}, {valid_preds.max():.3f}]")
        print(f"Targets shape: {valid_targets.shape}, unique values: {np.unique(valid_targets)}")
        return None

def plot_success_rate(metrics_dict, save_dir='plots'):
    """Plot success rate for different datasets."""
    print("Plotting success rate for different datasets...")
    output_file = os.path.join(save_dir, 'success_rate.svg')
    print(f"Saving plot to: {output_file}")
    plt.figure(figsize=(3.5, 2.5))
    
    # Prepare data for plotting
    datasets = list(metrics_dict.keys())
    x = np.arange(len(datasets))
    width = 0.25  # Reduced from 0.35 to accommodate three bars
    
    # Calculate success rate and error for each dataset
    success_rates = []
    max_success_rates = []
    all_pockets_success_rates = []
    success_errors = []
    max_success_errors = []
    all_pockets_success_errors = []
    
    for dataset in datasets:
        if metrics_dict[dataset]['success']:
            # Original success metric
            successes = np.array(metrics_dict[dataset]['success'])
            success_rate = np.mean(successes)
            n_samples = len(successes)
            success_error = np.std(successes) / np.sqrt(n_samples) if n_samples > 0 else 0
            success_rates.append(success_rate)
            success_errors.append(success_error)
            
            # Max-based success metric
            max_successes = np.array(metrics_dict[dataset]['max_success'])
            max_success_rate = np.mean(max_successes)
            max_success_error = np.std(max_successes) / np.sqrt(n_samples) if n_samples > 0 else 0
            max_success_rates.append(max_success_rate)
            max_success_errors.append(max_success_error)
            
            # All-pockets success metric
            all_pockets_successes = np.array(metrics_dict[dataset]['all_pockets_success'])
            all_pockets_success_rate = np.mean(all_pockets_successes)
            all_pockets_success_error = np.std(all_pockets_successes) / np.sqrt(n_samples) if n_samples > 0 else 0
            all_pockets_success_rates.append(all_pockets_success_rate)
            all_pockets_success_errors.append(all_pockets_success_error)
        else:
            success_rates.append(0)
            max_success_rates.append(0)
            all_pockets_success_rates.append(0)
            success_errors.append(0)
            max_success_errors.append(0)
            all_pockets_success_errors.append(0)
    
    # Create bar plot with error bars for all three metrics
    x = np.arange(len(datasets))
    width = 0.25
    
    # Plot max-based success rate (first)
    bars1 = plt.bar(x - width, max_success_rates, width, 
                    color='#43A3EF', edgecolor='black')
    plt.errorbar(x - width, max_success_rates, yerr=max_success_errors, 
                fmt='none', capsize=5, color='black')
    
    # Plot all-pockets success rate (second)
    bars2 = plt.bar(x, all_pockets_success_rates, width, 
                    color='#EF767B', edgecolor='black')
    plt.errorbar(x, all_pockets_success_rates, yerr=all_pockets_success_errors, 
                fmt='none', capsize=5, color='black')
    
    # Plot original success rate (third)
    bars3 = plt.bar(x + width, success_rates, width, 
                   color='white', edgecolor='black')
    plt.errorbar(x + width, success_rates, yerr=success_errors, 
                fmt='none', capsize=5, color='black')
    
    # Add value labels on top of bars
    for i, v in enumerate(max_success_rates):
        plt.text(i - width, v + 0.03, f'{v:.2f}', ha='center', fontsize=8)
    for i, v in enumerate(all_pockets_success_rates):
        plt.text(i, v + 0.03, f'{v:.2f}', ha='center', fontsize=8)
    for i, v in enumerate(success_rates):
        plt.text(i + width, v + 0.03, f'{v:.2f}', ha='center', fontsize=8)
    
    plt.xlabel('Dataset')
    plt.ylabel('Success Rate')
    plt.xticks(x, datasets)
    plt.ylim(0, 1.1)  # Set y-axis limit to accommodate labels
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Show and save plot
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_ratio_distributions(metrics_dict, save_dir='plots'):
    """Plot distribution of positive prediction ratios and true pocket ratios for COACH420 and Holo4K."""
    # First plot
    print("Plotting positive prediction ratio distribution...")
    output_file = os.path.join(save_dir, 'positive_ratio_distribution.svg')
    print(f"Saving plot to: {output_file}")
    plt.figure(figsize=(3.5, 2.5))
    
    # Calculate mean positive ratios
    coach_pos_ratio = np.mean([metrics_dict['COACH420']['iqr_ratio']])
    holo_pos_ratio = np.mean([metrics_dict['Holo4K']['iqr_ratio']])
    
    # Create density plots
    coach_data = metrics_dict['COACH420']['iqr_ratio']
    holo_data = metrics_dict['Holo4K']['iqr_ratio']
    
    # Calculate kernel density estimation
    coach_kde = np.histogram(coach_data, bins=30, density=True)
    holo_kde = np.histogram(holo_data, bins=30, density=True)
    
    # Plot curves with fills
    plt.plot((coach_kde[1][:-1] + coach_kde[1][1:]) / 2, coach_kde[0], color='#43A3EF', linewidth=2, label='COACH420')
    plt.fill_between((coach_kde[1][:-1] + coach_kde[1][1:]) / 2, coach_kde[0], alpha=0.3, color='#43A3EF')
    plt.plot((holo_kde[1][:-1] + holo_kde[1][1:]) / 2, holo_kde[0], color='#EF767B', linewidth=2, label='Holo4K')
    plt.fill_between((holo_kde[1][:-1] + holo_kde[1][1:]) / 2, holo_kde[0], alpha=0.3, color='#EF767B')
    
    # Add vertical lines for means
    plt.axvline(x=coach_pos_ratio, color='#43A3EF', linestyle='--', alpha=0.8, linewidth=2)
    plt.axvline(x=holo_pos_ratio, color='#EF767B', linestyle='--', alpha=0.8, linewidth=2)
    
    plt.xlabel('Ratio of Residues Above IQR Threshold')
    plt.ylabel('Density')
    plt.legend(loc='upper right', frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show plot
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    plt.close()

    # Second plot
    print("Plotting true pocket ratio distribution...")
    output_file = os.path.join(save_dir, 'true_ratio_distribution.svg')
    print(f"Saving plot to: {output_file}")
    plt.figure(figsize=(3.5, 2.5))
    
    # Calculate mean true ratios
    coach_true_ratio = np.mean([metrics_dict['COACH420']['true_ratio']])
    holo_true_ratio = np.mean([metrics_dict['Holo4K']['true_ratio']])
    
    # Calculate kernel density estimation
    coach_data = metrics_dict['COACH420']['true_ratio']
    holo_data = metrics_dict['Holo4K']['true_ratio']
    
    coach_kde = np.histogram(coach_data, bins=30, density=True)
    holo_kde = np.histogram(holo_data, bins=30, density=True)
    
    # Plot curves with fills
    plt.plot((coach_kde[1][:-1] + coach_kde[1][1:]) / 2, coach_kde[0], color='#43A3EF', linewidth=2, label='COACH420')
    plt.fill_between((coach_kde[1][:-1] + coach_kde[1][1:]) / 2, coach_kde[0], alpha=0.3, color='#43A3EF')
    plt.plot((holo_kde[1][:-1] + holo_kde[1][1:]) / 2, holo_kde[0], color='#EF767B', linewidth=2, label='Holo4K')
    plt.fill_between((holo_kde[1][:-1] + holo_kde[1][1:]) / 2, holo_kde[0], alpha=0.3, color='#EF767B')
    
    # Add vertical lines for means
    plt.axvline(x=coach_true_ratio, color='#43A3EF', linestyle='--', alpha=0.8, linewidth=2)
    plt.axvline(x=holo_true_ratio, color='#EF767B', linestyle='--', alpha=0.8, linewidth=2)
    
    plt.xlabel('True Pocket Ratio')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show plot
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_probability_distributions(metrics_dict, save_dir='plots'):
    """Plot probability-related distributions."""
    # First plot
    # print("Plotting distribution of max probability at true pockets...")
    # output_file = os.path.join(save_dir, 'max_prob_at_true.svg')
    # print(f"Saving plot to: {output_file}")
    # plt.figure(figsize=(3.5, 2.5))
    
    # # Plot histograms for both datasets
    # plt.hist(metrics_dict['COACH420']['max_prob_at_true'], bins=50, alpha=0.7,
    #          label=f'COACH420 (mean={np.mean(metrics_dict["COACH420"]["max_prob_at_true"]):.3f})', 
    #          color='#43A3EF', edgecolor='black', density=True)
    # plt.hist(metrics_dict['Holo4K']['max_prob_at_true'], bins=50, alpha=0.7,
    #          label=f'Holo4K (mean={np.mean(metrics_dict["Holo4K"]["max_prob_at_true"]):.3f})', 
    #          color='#EF767B', edgecolor='black', density=True)
    
    # plt.xlabel('Max Probability at True Pockets')
    # plt.ylabel('Density')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    
    # # Show and save plot
    # plt.savefig(output_file, bbox_inches='tight')
    # plt.show()
    # plt.close()

    # Second plot
    print("Plotting distribution of probability differences...")
    output_file = os.path.join(save_dir, 'diff_max_prob_at_true_overall_max_prob.svg')
    print(f"Saving plot to: {output_file}")
    plt.figure(figsize=(3.5, 2.5))
    
    # Calculate differences for both datasets
    coach_diff = np.array(metrics_dict['COACH420']['max_prob_at_true']) - np.array(metrics_dict['COACH420']['overall_max_prob'])
    holo_diff = np.array(metrics_dict['Holo4K']['max_prob_at_true']) - np.array(metrics_dict['Holo4K']['overall_max_prob'])
    
    plt.hist(coach_diff, bins=50, alpha=0.7,
             label=f'COACH420 (mean={np.mean(coach_diff):.3f})', 
             color='#43A3EF', edgecolor='black', density=True)
    plt.hist(holo_diff, bins=50, alpha=0.7,
             label=f'Holo4K (mean={np.mean(holo_diff):.3f})', 
             color='#EF767B', edgecolor='black', density=True)
    
    plt.xlabel('Difference between max_probs_at_true and overall_max_probs')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Show and save plot
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_pocket_statistics(metrics_dict, save_dir='plots'):
    """Plot pocket statistics for different datasets."""
    print("Plotting pocket statistics...")
    output_file = os.path.join(save_dir, 'pocket_statistics.svg')
    print(f"Saving plot to: {output_file}")
    plt.figure(figsize=(3.5, 2.5))
    
    datasets = list(metrics_dict.keys())
    pocket_metrics = ['n_pred_pockets', 'n_true_pockets', 'n_overlap']
    pocket_data = {dataset: [] for dataset in datasets}
    
    for dataset in datasets:
        for metric in pocket_metrics:
            if metrics_dict[dataset][metric]:
                pocket_data[dataset].append(np.mean(metrics_dict[dataset][metric]))
            else:
                pocket_data[dataset].append(0)
    
    # Create grouped bar plot
    x = np.arange(len(datasets))
    width = 0.25
    
    colors = ['#43A3EF', '#EF767B', '#43A3EF']
    for i, metric in enumerate(pocket_metrics):
        values = [pocket_data[dataset][i] for dataset in datasets]
        bars = plt.bar(x + i*width, values, width, 
                      label=metric.replace('n_', '').replace('_', ' ').title(), 
                      color=colors[i])
        # Add black edges to all bars
        for bar in bars:
            bar.set_edgecolor('black')
            bar.set_linewidth(1)
    
    plt.xlabel('Dataset')
    plt.ylabel('Count')
    plt.xticks(x + width, datasets)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Show and save plot
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_nonzero_distribution(metrics_dict, save_dir='plots'):
    """Plot distribution of non-zero prediction values."""
    print("Plotting distribution of non-zero prediction values...")
    output_file = os.path.join(save_dir, 'nonzero_prediction_distribution.svg')
    print(f"Saving plot to: {output_file}")
    plt.figure(figsize=(3.5, 2.5))
    
    # Concatenate all non-zero predictions for each dataset
    coach_nonzero = np.concatenate(metrics_dict['COACH420']['nonzero_preds'])
    holo_nonzero = np.concatenate(metrics_dict['Holo4K']['nonzero_preds'])
    
    # Calculate means
    coach_mean = np.mean(coach_nonzero)
    holo_mean = np.mean(holo_nonzero)
    
    # Calculate kernel density estimation
    coach_kde = np.histogram(coach_nonzero, bins=50, density=True)
    holo_kde = np.histogram(holo_nonzero, bins=50, density=True)
    
    # Plot curves with fills
    plt.plot((coach_kde[1][:-1] + coach_kde[1][1:]) / 2, coach_kde[0], color='#43A3EF', linewidth=2, label='COACH420')
    plt.fill_between((coach_kde[1][:-1] + coach_kde[1][1:]) / 2, coach_kde[0], alpha=0.3, color='#43A3EF')
    plt.plot((holo_kde[1][:-1] + holo_kde[1][1:]) / 2, holo_kde[0], color='#EF767B', linewidth=2, label='Holo4K')
    plt.fill_between((holo_kde[1][:-1] + holo_kde[1][1:]) / 2, holo_kde[0], alpha=0.3, color='#EF767B')
    
    # Add vertical lines for means
    plt.axvline(x=coach_mean, color='#43A3EF', linestyle='--', alpha=0.8, linewidth=2)
    plt.axvline(x=holo_mean, color='#EF767B', linestyle='--', alpha=0.8, linewidth=2)
    
    plt.xlabel('Non-zero Prediction Values')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show plot
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_threshold_ratios(metrics_dict, save_dir='plots'):
    """Plot ratio of residues above threshold vs threshold value."""
    print("Plotting ratio of residues above threshold vs threshold value...")
    output_file = os.path.join(save_dir, 'threshold_ratios.svg')
    print(f"Saving plot to: {output_file}")
    plt.figure(figsize=(3.5, 2.5))
    
    # Get thresholds
    thresholds = metrics_dict['COACH420']['thresholds'][0]
    width = 0.35  # Width of the bars
    
    # Calculate mean ratios for each threshold
    ratios = metrics_dict['COACH420']['threshold_ratios']
    n_ratios = len(ratios)
    coach_ratios = [np.mean([ratios[j][i] for j in range(n_ratios)]) for i in range(len(thresholds))]
    
    ratios = metrics_dict['Holo4K']['threshold_ratios']
    n_ratios = len(ratios)
    holo_ratios = [np.mean([ratios[j][i] for j in range(n_ratios)]) for i in range(len(thresholds))]
    
    # Create bar plot
    x = np.arange(len(thresholds))
    plt.bar(x - width/2, coach_ratios, width, color='#43A3EF', edgecolor='black', label='COACH420')
    plt.bar(x + width/2, holo_ratios, width, color='#EF767B', edgecolor='black', label='Holo4K')
    
    plt.xlabel('Threshold')
    plt.ylabel('Ratio')
    xticks = [i for i in range(0, len(thresholds), 2)]
    plt.xticks(xticks, [f'{t:.1f}' for t in thresholds[xticks]])
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show plot
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_classification_metrics(metrics_dict, save_dir='plots'):
    """Plot ROC curve and Precision-Recall curve for pocket prediction."""
    print("Plotting classification metrics...")
    
    # Plot ROC curve
    plt.figure(figsize=(3.5, 2.5))
    print("\nROC curve metrics:")
    for dataset, color in [('COACH420', '#43A3EF'), ('Holo4K', '#EF767B')]:
        all_preds = []
        all_targets = []
        # Collect all predictions and targets (already masked in evaluate_predictions)
        for i in range(len(metrics_dict[dataset]['predictions'])):
            all_preds.extend(metrics_dict[dataset]['predictions'][i])
            all_targets.extend(metrics_dict[dataset]['is_pocket'][i])
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(all_targets, all_preds)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, linewidth=2, label=dataset)
        print(f"{dataset} AUC: {roc_auc:.3f}")
    
    # ROC curve settings
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save ROC plot
    output_file = os.path.join(save_dir, 'roc_curve.svg')
    print(f"Saving ROC plot to: {output_file}")
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Plot PR curve
    plt.figure(figsize=(3.5, 2.5))
    print("\nPR curve metrics:")
    for dataset, color in [('COACH420', '#43A3EF'), ('Holo4K', '#EF767B')]:
        all_preds = []
        all_targets = []
        # Collect all predictions and targets (already masked in evaluate_predictions)
        for i in range(len(metrics_dict[dataset]['predictions'])):
            all_preds.extend(metrics_dict[dataset]['predictions'][i])
            all_targets.extend(metrics_dict[dataset]['is_pocket'][i])
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Calculate random baseline for PR curve using all targets
        if dataset == 'COACH420':
            n_samples = 1000
            random_preds = np.random.rand(len(all_targets))
            precision_random, recall_random, _ = precision_recall_curve(all_targets, random_preds)
            ap_random = average_precision_score(all_targets, random_preds)
            plt.plot(recall_random, precision_random, 'k--', alpha=0.5, label='Random')
            print(f"Random AP: {ap_random:.3f}")
        
        # Plot PR curve
        precision, recall, _ = precision_recall_curve(all_targets, all_preds)
        ap = average_precision_score(all_targets, all_preds)
        plt.plot(recall, precision, color=color, linewidth=2, label=dataset)
        print(f"{dataset} AP: {ap:.3f}")
    
    # PR curve settings
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='upper right', frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save PR plot
    output_file = os.path.join(save_dir, 'pr_curve.svg')
    print(f"Saving PR plot to: {output_file}")
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_metrics_distributions(metrics_dict, save_dir='plots'):
    """Plot distributions of ROC AUC and AP scores."""
    print("Plotting metrics distributions...")
    
    # Plot AUC distribution
    plt.figure(figsize=(3.5, 2.5))
    for dataset, color in [('COACH420', '#43A3EF'), ('Holo4K', '#EF767B')]:
        aucs = metrics_dict[dataset]['roc_auc']
        # Plot AUC distribution
        kde = np.histogram(aucs, bins=30, density=True)
        plt.plot((kde[1][:-1] + kde[1][1:]) / 2, kde[0], color=color, linewidth=2,
                label=f'{dataset}')
    
    plt.xlabel('ROC AUC')
    plt.ylabel('Density')
    plt.legend(loc='upper left', frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save AUC distribution plot
    output_file = os.path.join(save_dir, 'auc_distribution.svg')
    print(f"Saving AUC distribution plot to: {output_file}")
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Plot AP distribution
    plt.figure(figsize=(3.5, 2.5))
    for dataset, color in [('COACH420', '#43A3EF'), ('Holo4K', '#EF767B')]:
        aps = metrics_dict[dataset]['ap']
        # Plot AP distribution
        kde = np.histogram(aps, bins=30, density=True)
        plt.plot((kde[1][:-1] + kde[1][1:]) / 2, kde[0], color=color, linewidth=2,
                label=f'{dataset}')
    
    plt.xlabel('Average Precision')
    plt.ylabel('Density')
    plt.legend(loc='upper right', frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save AP distribution plot
    output_file = os.path.join(save_dir, 'ap_distribution.svg')
    print(f"Saving AP distribution plot to: {output_file}")
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_ratio_distribution(metrics_dict, save_dir='plots'):
    """Plot distribution of ratios of residues above IQR threshold."""
    print("Plotting distribution of ratios above IQR threshold...")
    output_file = os.path.join(save_dir, 'ratio_distribution.svg')
    print(f"Saving plot to: {output_file}")
    plt.figure(figsize=(3.5, 2.5))
    
    # Calculate ratios for each dataset
    for dataset, color in [('COACH420', '#43A3EF'), ('Holo4K', '#EF767B')]:
        # Get all predictions
        all_preds = metrics_dict[dataset]['predictions']
        ratios = []
        
        # Calculate ratio for each protein
        for preds in all_preds:
            # Calculate IQR threshold
            q1 = np.percentile(preds, 25)
            q3 = np.percentile(preds, 75)
            iqr = q3 - q1
            iqr_threshold = q3 + 1.5 * iqr
            
            # Calculate ratio of residues above threshold
            ratio = np.mean(preds > iqr_threshold)
            ratios.append(ratio)
        
        # Calculate kernel density estimation
        kde = np.histogram(ratios, bins=30, density=True)
        
        # Plot curves with fills
        plt.plot((kde[1][:-1] + kde[1][1:]) / 2, kde[0], color=color, linewidth=2, label=dataset)
        plt.fill_between((kde[1][:-1] + kde[1][1:]) / 2, kde[0], alpha=0.3, color=color)
        
        # Add vertical lines for means
        mean_ratio = np.mean(ratios)
        plt.axvline(x=mean_ratio, color=color, linestyle='--', alpha=0.8, linewidth=2)
        print(f"{dataset} mean ratio: {mean_ratio:.3f}")
    
    plt.xlabel('Ratio of Residues Above IQR Threshold')
    plt.ylabel('Density')
    plt.legend(loc='upper left', frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show plot
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_iqr_threshold_distribution(metrics_dict, save_dir='plots'):
    """Plot distribution of IQR thresholds."""
    print("Plotting IQR threshold distribution...")
    output_file = os.path.join(save_dir, 'iqr_threshold_distribution.svg')
    print(f"Saving plot to: {output_file}")
    plt.figure(figsize=(3.5, 2.5))
    
    # Plot distributions for both datasets
    for dataset, color in [('COACH420', '#43A3EF'), ('Holo4K', '#EF767B')]:
        iqr_thresholds = metrics_dict[dataset]['iqr_threshold']
        
        # Calculate kernel density estimation
        kde = np.histogram(iqr_thresholds, bins=30, density=True)
        
        # Plot curves with fills
        plt.plot((kde[1][:-1] + kde[1][1:]) / 2, kde[0], color=color, linewidth=2, label=dataset)
        plt.fill_between((kde[1][:-1] + kde[1][1:]) / 2, kde[0], alpha=0.3, color=color)
        
        # Add vertical lines for means
        mean_threshold = np.mean(iqr_thresholds)
        plt.axvline(x=mean_threshold, color=color, linestyle='--', alpha=0.8, linewidth=2)
        print(f"{dataset} mean IQR threshold: {mean_threshold:.3f}")
    
    plt.xlabel('IQR Threshold')
    plt.ylabel('Density')
    plt.legend(loc='upper right', frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show plot
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_non_pocket_pred_distribution(metrics_dict, save_dir='plots'):
    """Plot distribution of prediction values for non-pocket residues (is_pocket=0) using box plot and swarm plot."""
    print("Plotting distribution of prediction values for non-pocket residues...")
    output_file = os.path.join(save_dir, 'non_pocket_pred_distribution.svg')
    print(f"Saving plot to: {output_file}")
    plt.figure(figsize=(3.5, 2.5))
    
    # Collect data for both datasets
    data = []
    labels = []
    colors = []
    swarm_x = []
    swarm_y = []
    for i, (dataset, color) in enumerate([('COACH420', '#43A3EF'), ('Holo4K', '#EF767B')]):
        all_non_pocket_preds = []
        
        # Collect predictions for non-pocket residues
        for preds, is_pocket in zip(metrics_dict[dataset]['predictions'], metrics_dict[dataset]['is_pocket']):
            non_pocket_preds = preds[is_pocket == 0]
            all_non_pocket_preds.extend(non_pocket_preds)
        
        # Filter data for swarm plot (only values <= 0.1)
        filtered_preds = [p for p in all_non_pocket_preds if p <= 0.1]
        # Randomly sample 1000 points if there are too many
        if len(filtered_preds) > 1000:
            filtered_preds = np.random.choice(filtered_preds, 1000, replace=False)
        
        # Add random jitter to x positions based on y values
        # More points at a y value = wider spread
        y_values = np.array(filtered_preds)
        n_points = len(y_values)
        
        # Calculate width of jitter based on density
        y_bins = np.linspace(0, 0.1, 50)  # 50 bins
        hist, _ = np.histogram(y_values, bins=y_bins)
        max_count = np.max(hist)
        
        # For each y value, calculate its bin and corresponding width
        y_digitized = np.digitize(y_values, y_bins)
        widths = []
        for y_idx in y_digitized:
            count_in_bin = hist[min(y_idx-1, len(hist)-1)]
            width = 0.4 * (count_in_bin / max_count)  # max width is 0.4
            widths.append(width)
        
        # Generate x positions with varying spread
        x_positions = []
        for width in widths:
            x_positions.append(i + 1 + np.random.uniform(-width, width))
        
        swarm_y.extend(filtered_preds)
        swarm_x.extend(x_positions)
        
        data.append(all_non_pocket_preds)
        labels.append(dataset)
        colors.append(color)
        
        # Print statistics
        print(f"{dataset} statistics:")
        print(f"  Mean: {np.mean(all_non_pocket_preds):.3f}")
        print(f"  Median: {np.median(all_non_pocket_preds):.3f}")
        print(f"  Std: {np.std(all_non_pocket_preds):.3f}")
        print(f"  Number of residues: {len(all_non_pocket_preds)}")
    
    # Create box plot without outliers
    bp = plt.boxplot(data, labels=labels, patch_artist=True, medianprops=dict(color='black', linewidth=1.5),
                    showfliers=False)
    
    # Color the boxes
    for box, color in zip(bp['boxes'], colors):
        box.set_facecolor(color)
        box.set_alpha(0.5)
        box.set_edgecolor('black')
    
    # Add swarm plot
    plt.scatter(swarm_x, swarm_y, c='black', alpha=0.2, s=3, edgecolors='none')
    
    plt.ylabel('Prediction Value')
    plt.ylim(0, 0.1)  # Set y-axis range
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show plot
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_iqr_vs_size(metrics_dict, save_dir='plots'):
    """Plot relationship between IQR threshold and protein size."""
    print("Plotting IQR threshold vs protein size...")
    output_file = os.path.join(save_dir, 'iqr_vs_size.svg')
    print(f"Saving plot to: {output_file}")
    plt.figure(figsize=(3.5, 2.5))
    
    # Plot scatter for both datasets
    for dataset, color in [('COACH420', '#43A3EF'), ('Holo4K', '#EF767B')]:
        # Get protein sizes and IQR thresholds
        sizes = [len(preds) for preds in metrics_dict[dataset]['predictions']]
        iqr_thresholds = metrics_dict[dataset]['iqr_threshold']
        
        # Plot scatter points with smaller markers and no edge color
        plt.scatter(sizes, iqr_thresholds, c=color, alpha=0.5, label=dataset, s=10, edgecolors='none')
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(sizes, iqr_thresholds)[0,1]
        print(f"{dataset} correlation coefficient: {correlation:.3f}")
        print(f"{dataset} size range: [{min(sizes)}, {max(sizes)}]")
        print(f"{dataset} mean size: {np.mean(sizes):.1f}")
    
    plt.xlabel('Protein Size (Number of Residues)')
    plt.ylabel('IQR Threshold')
    plt.legend(loc='upper right', frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show plot
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_pocket_size_vs_protein_size(metrics_dict, save_dir='plots'):
    """Plot relationship between number of pocket residues and protein size."""
    print("Plotting number of pocket residues vs protein size...")
    output_file = os.path.join(save_dir, 'pocket_size_vs_protein_size.svg')
    print(f"Saving plot to: {output_file}")
    plt.figure(figsize=(3.5, 2.5))
    
    # Plot scatter for both datasets
    for dataset, color in [('COACH420', '#43A3EF'), ('Holo4K', '#EF767B')]:
        # Get protein sizes and number of pocket residues
        protein_sizes = [len(preds) for preds in metrics_dict[dataset]['predictions']]
        pocket_sizes = [np.sum(is_pocket) for is_pocket in metrics_dict[dataset]['is_pocket']]
        
        # Plot scatter points with smaller markers and no edge color
        plt.scatter(protein_sizes, pocket_sizes, c=color, alpha=0.5, label=dataset, s=10, edgecolors='none')
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(protein_sizes, pocket_sizes)[0,1]
        print(f"{dataset} correlation coefficient: {correlation:.3f}")
        print(f"{dataset} mean pocket size: {np.mean(pocket_sizes):.1f}")
        print(f"{dataset} mean pocket ratio: {np.mean(np.array(pocket_sizes) / np.array(protein_sizes)):.3f}")
    
    plt.xlabel('Protein Size (Number of Residues)')
    plt.ylabel('Number of Pocket Residues')
    plt.legend(loc='upper right', frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show plot
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_iqr_ratio_vs_size(metrics_dict, save_dir='plots'):
    """Plot relationship between ratio of residues above IQR threshold and protein size."""
    print("Plotting ratio of residues above IQR threshold vs protein size...")
    output_file = os.path.join(save_dir, 'iqr_ratio_vs_size.svg')
    print(f"Saving plot to: {output_file}")
    plt.figure(figsize=(3.5, 2.5))
    
    # Plot scatter for both datasets
    for dataset, color in [('COACH420', '#43A3EF'), ('Holo4K', '#EF767B')]:
        # Get protein sizes and IQR ratios
        sizes = [len(preds) for preds in metrics_dict[dataset]['predictions']]
        iqr_ratios = metrics_dict[dataset]['iqr_ratio']
        
        # Plot scatter points with smaller markers and no edge color
        plt.scatter(sizes, iqr_ratios, c=color, alpha=0.5, label=dataset, s=10, edgecolors='none')
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(sizes, iqr_ratios)[0,1]
        print(f"{dataset} correlation coefficient: {correlation:.3f}")
        print(f"{dataset} mean ratio: {np.mean(iqr_ratios):.3f}")
        print(f"{dataset} std ratio: {np.std(iqr_ratios):.3f}")
    
    plt.xlabel('Protein Size (Number of Residues)')
    plt.ylabel('Ratio of Residues Above IQR Threshold')
    plt.legend(loc='upper right', frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show plot
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_confusion_matrix_metrics(metrics_dict, save_dir='plots'):
    """Plot confusion matrix based metrics for both threshold-based and IQR-based predictions."""
    print("Plotting confusion matrix based metrics...")
    output_file = os.path.join(save_dir, 'confusion_matrix_metrics.svg')
    print(f"Saving plot to: {output_file}")
    plt.figure(figsize=(5, 3.5))  # Made figure wider to accommodate more metrics
    
    # Metrics to plot
    metric_names = [
        'precision', 'npv', 'recall', 'specificity', 'accuracy', 
        'mcc', 'f1_score'
    ]
    
    # Metrics display names
    metric_display = {
        'precision': 'Precision',
        'recall': 'Recall',
        'specificity': 'Specificity',
        'accuracy': 'Accuracy',
        'mcc': 'MCC',
        'f1_score': 'F1',
        'npv': 'NPV'
    }
    
    # Calculate mean and std for each metric and dataset
    datasets = ['COACH420', 'Holo4K']
    edge_colors = ['#43A3EF', '#EF767B']
    face_colors = ['#A8D5F7', '#F7BBBE']  # Lighter versions of the edge colors
    x = np.arange(len(metric_names))
    width = 0.15  # Reduced width to accommodate 4 bars
    
    for i, (dataset, edge_color, face_color) in enumerate(zip(datasets, edge_colors, face_colors)):
        # Threshold-based metrics
        means = []
        stds = []
        for metric in metric_names:
            values = metrics_dict[dataset][metric]
            mean = np.mean(values)
            std = np.std(values) / np.sqrt(len(values))  # Standard error
            means.append(mean)
            stds.append(std)
            print(f"{dataset} threshold {metric}: {mean:.3f} ± {std:.3f}")
        
        # Create threshold-based bars
        plt.bar(x + i*width*2, means, width, label=f'{dataset} (Threshold)',
                color=face_color, edgecolor=edge_color, linewidth=1,
                yerr=stds, capsize=3, error_kw=dict(ecolor=edge_color, capthick=1))
        
        # IQR-based metrics
        means_iqr = []
        stds_iqr = []
        for metric in metric_names:
            values = metrics_dict[dataset][f'{metric}_iqr']
            mean = np.mean(values)
            std = np.std(values) / np.sqrt(len(values))  # Standard error
            means_iqr.append(mean)
            stds_iqr.append(std)
            print(f"{dataset} IQR {metric}: {mean:.3f} ± {std:.3f}")
        
        # Create IQR-based bars
        plt.bar(x + i*width*2 + width, means_iqr, width, label=f'{dataset} (IQR)',
                color='white', edgecolor=edge_color, linewidth=1,
                yerr=stds_iqr, capsize=3, error_kw=dict(ecolor=edge_color, capthick=1), hatch='//')
    
    # Customize plot
    # plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.xticks(x + width*1.5, [metric_display[name] for name in metric_names], rotation=45)
    plt.legend(bbox_to_anchor=(0.5, 1.15), loc='center', ncol=2, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show plot
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_confusion_matrix_metrics_all_pockets(metrics_dict, save_dir='plots'):
    """Plot confusion matrix based metrics against all_pockets for both threshold-based and IQR-based predictions."""
    print("Plotting confusion matrix metrics against all pockets...")
    output_file = os.path.join(save_dir, 'confusion_matrix_metrics_all_pockets.svg')
    print(f"Saving plot to: {output_file}")
    plt.figure(figsize=(5, 3.5))  # Made figure wider to accommodate more metrics
    
    # Metrics to plot
    metric_names = [
        'precision', 'npv', 'recall', 'specificity', 'accuracy', 
        'mcc', 'f1_score'
    ]
    
    # Metrics display names
    metric_display = {
        'precision': 'Precision',
        'recall': 'Recall',
        'specificity': 'Specificity',
        'accuracy': 'Accuracy',
        'mcc': 'MCC',
        'f1_score': 'F1',
        'npv': 'NPV'
    }
    
    # Calculate mean and std for each metric and dataset
    datasets = ['COACH420', 'Holo4K']
    edge_colors = ['#43A3EF', '#EF767B']
    face_colors = ['#A8D5F7', '#F7BBBE']  # Lighter versions of the edge colors
    x = np.arange(len(metric_names))
    width = 0.15  # Reduced width to accommodate 4 bars
    
    for i, (dataset, edge_color, face_color) in enumerate(zip(datasets, edge_colors, face_colors)):
        # Skip if all_pockets metrics are not available
        if f'precision_all' not in metrics_dict[dataset]:
            print(f"Warning: all_pockets metrics not available for {dataset}")
            continue
            
        # Threshold-based metrics
        means = []
        stds = []
        for metric in metric_names:
            values = metrics_dict[dataset][f'{metric}_all']
            mean = np.mean(values)
            std = np.std(values) / np.sqrt(len(values))  # Standard error
            means.append(mean)
            stds.append(std)
            print(f"{dataset} threshold {metric}_all: {mean:.3f} ± {std:.3f}")
        
        # Create threshold-based bars
        plt.bar(x + i*width*2, means, width, label=f'{dataset} (Threshold)',
                color=face_color, edgecolor=edge_color, linewidth=1,
                yerr=stds, capsize=3, error_kw=dict(ecolor=edge_color, capthick=1))
        
        # IQR-based metrics
        means_iqr = []
        stds_iqr = []
        for metric in metric_names:
            values = metrics_dict[dataset][f'{metric}_all_iqr']
            mean = np.mean(values)
            std = np.std(values) / np.sqrt(len(values))  # Standard error
            means_iqr.append(mean)
            stds_iqr.append(std)
            print(f"{dataset} IQR {metric}_all: {mean:.3f} ± {std:.3f}")
        
        # Create IQR-based bars
        plt.bar(x + i*width*2 + width, means_iqr, width, label=f'{dataset} (IQR)',
                color='white', edgecolor=edge_color, linewidth=1,
                yerr=stds_iqr, capsize=3, error_kw=dict(ecolor=edge_color, capthick=1), hatch='//')
    
    # Customize plot
    # plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.xticks(x + width*1.5, [metric_display[name] for name in metric_names], rotation=45)
    plt.legend(bbox_to_anchor=(0.5, 1.15), loc='center', ncol=2, frameon=False)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show plot
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()
    plt.close()

def analyze_results():
    """Analyze prediction results from the database."""
    try:
        # Get results from database
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT r.id, r.pocket_pred, d.pname, d.in_coach420, d.in_holo4k,
                       d.is_pocket, d.protein_mask, p.all_pockets, d.lname
                FROM moad_test_results r
                JOIN processed_datasets d ON r.id = d.id
                LEFT JOIN proteins p ON d.pname = p.name
                WHERE d.split = 'test'
            """)
            results = cursor.fetchall()
            cursor.close()
        
        if not results:
            print("No results found in the database.")
            return None
        
        # Initialize results dictionary
        analysis = {
            'COACH420': {}, 'Holo4K': {}, 'Both': {}
        }
        
        # Process results
        # threshold = 0.8
        threshold = 0.06
        print("Threshold: ", threshold)
        for row in results:
            id_, pocket_pred, pname, is_coach420, is_holo4k, is_pocket, protein_mask, all_pockets, ligand_name = row
            pred = pickle.loads(pocket_pred).squeeze()
            is_pocket = pickle.loads(is_pocket).squeeze()
            protein_mask = pickle.loads(protein_mask).squeeze()
            all_pockets = pickle.loads(all_pockets) if all_pockets is not None else None

            if pred.shape[0] != is_pocket.shape[0] or pred.shape[0] != protein_mask.shape[0]:
                print(f"Error: protein {pname} has different shapes: pred.shape={pred.shape}, is_pocket.shape={is_pocket.shape}, protein_mask.shape={protein_mask.shape}")
                continue
            
            # Calculate metrics
            try:
                metrics = evaluate_predictions(pred, is_pocket, protein_mask, all_pockets, threshold)
                if metrics is None:
                    continue
                
                # Store results by dataset
                if is_coach420:
                    for metric, value in metrics.items():
                        analysis['COACH420'].setdefault(metric, []).append(value)
                    analysis['COACH420'].setdefault('names', []).append(pname)
                    analysis['COACH420'].setdefault('ligand_names', []).append(ligand_name)
                if is_holo4k:
                    for metric, value in metrics.items():
                        analysis['Holo4K'].setdefault(metric, []).append(value)
                    analysis['Holo4K'].setdefault('names', []).append(pname)
                    analysis['Holo4K'].setdefault('ligand_names', []).append(ligand_name)
                for metric, value in metrics.items():
                    analysis['Both'].setdefault(metric, []).append(value)
                analysis['Both'].setdefault('names', []).append(pname)
                analysis['Both'].setdefault('ligand_names', []).append(ligand_name)
            except Exception as e:
                print(f"Error processing protein {pname}: {str(e)}")
                raise e
                
        return analysis
            
    except Exception as e:
        print(f"Error in analyze_results: {str(e)}")
        raise e

def print_random_case(metrics_dict, cb):
    """Print a random protein with AP score lower than the threshold."""
    print(f"\nLooking for proteins with {cb.__name__}...")
    
    # Collect all low AP cases
    cases = []
    for dataset in ['COACH420', 'Holo4K']:
        metrics = metrics_dict[dataset]
        keys = list(metrics.keys())
        n = len(metrics['names'])
        for i in range(n):
            metric = {key: metrics[key][i] for key in keys}
            metric['dataset'] = dataset
            if cb(metric):
                cases.append(metric)

    if not cases:
        print(f"No proteins found with {cb.__name__}.")
        return
    
    # Randomly select one case
    idx = np.random.randint(len(cases))
    selected = cases[idx]
    for key, value in selected.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Analyze results and plot metrics
    metrics = analyze_results()
    
    save_dir = 'plots'
    os.makedirs(save_dir, exist_ok=True)  # Create output directory if it doesn't exist
    
    # plot_success_rate(metrics, save_dir)
    # plot_confusion_matrix_metrics(metrics, save_dir)
    # plot_confusion_matrix_metrics_all_pockets(metrics, save_dir)
    # plot_ratio_distributions(metrics, save_dir)
    # plot_nonzero_distribution(metrics, save_dir)
    # plot_probability_distributions(metrics, save_dir)
    # plot_pocket_statistics(metrics, save_dir)
    # plot_threshold_ratios(metrics, save_dir)
    plot_classification_metrics(metrics, save_dir)
    # plot_metrics_distributions(metrics, save_dir)
    # plot_ratio_distribution(metrics, save_dir)
    # plot_iqr_threshold_distribution(metrics, save_dir)
    # plot_non_pocket_pred_distribution(metrics, save_dir)
    # plot_iqr_vs_size(metrics, save_dir)
    # plot_pocket_size_vs_protein_size(metrics, save_dir)
    # plot_iqr_ratio_vs_size(metrics, save_dir)
    
    # print_random_case(metrics, lambda metric: metric['ap'] < 0.1)
    # print_random_case(metrics, lambda metric: metric['ap'] > 0.9)
    # print_random_case(metrics, lambda metric: metric['iqr_threshold'] > 0.7)

# %%
