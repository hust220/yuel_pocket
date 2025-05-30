#%%

# read datasets/geom_test.pt and test_results/geom_test_predictions.pt
# compare edge_pred and edge_true
# print the accuracy, precision, recall, f1 score

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from src import const
import matplotlib.pyplot as plt
import json

groups = {
    # 'geom': ['geom_test', 'geom_test_predictions'],
    # 'geom_bonds': ['geom_test_bonds', 'geom_test_predictions_bonds'],
    # 'geom_kekulized': ['geom_kekulized_test', 'geom_kekulized_test_predictions'],
    # 'geom_kekulized_bonds': ['geom_kekulized_test_bonds', 'geom_kekulized_test_predictions_bonds'],
    # 'geom_sanitized': ['geom_sanitized_test', 'geom_sanitized_test_predictions'],
    # 'geom_sanitized_bonds': ['geom_sanitized_test_bonds', 'geom_sanitized_test_predictions_bonds'],
    'geom_sanitized_noise_0_2': ['geom_sanitized_test_noise_0_2', 'geom_sanitized_test_noise_0_2_predictions'],
}

def load_data(group):
    ground_truth = torch.load(f'datasets/{group[0]}.pt', map_location=torch.device('cpu'))
    predictions = torch.load(f'test_results/{group[1]}.pt', map_location=torch.device('cpu'))

    edge_preds = []
    for i in range(len(predictions)):
        for j in range(predictions[i]['edge_pred'].shape[0]):
            edge_preds.append(predictions[i]['edge_pred'][j].squeeze())

    edge_masks = []
    for i in range(len(predictions)):
        for j in range(predictions[i]['edge_mask'].shape[0]):
            edge_masks.append(predictions[i]['edge_mask'][j].squeeze())

    edge_trues = []
    for i in range(len(ground_truth)):
        edge_trues.append(ground_truth[i]['bond_orders'])

    return edge_preds, edge_masks, edge_trues

# Initialize lists to store metrics
def calculate_metrics(edge_preds, edge_masks, edge_trues):
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    # Process each molecule
    print("Calculating metrics...")
    for i in tqdm(range(len(edge_preds))):
        edge_pred = edge_preds[i]
        edge_mask = edge_masks[i]
        edge_true = edge_trues[i]

        # Convert to numpy arrays for sklearn metrics
        # First, get the indices where edge_mask is 1 (valid edges)
        valid_indices = torch.where(edge_mask.squeeze() == 1)[0]
        
        if len(valid_indices) == 0:
            continue
        
        # Get the predicted and true bond orders for valid edges
        pred_bond_types = torch.argmax(edge_pred.squeeze(), dim=1)[valid_indices].cpu().numpy()
        true_bond_types = torch.argmax(edge_true.squeeze(), dim=1)[valid_indices].cpu().numpy()
        # print(pred_bond_types)
        # print(true_bond_types)

        # Calculate metrics
        # 9 means no bond, 0 means single, 1 means double, 2 means triple, 3 means aromatic
        accuracy = accuracy_score(true_bond_types, pred_bond_types)
        precision = precision_score(true_bond_types, pred_bond_types, average='weighted', zero_division=0)
        recall = recall_score(true_bond_types, pred_bond_types, average='weighted', zero_division=0)
        f1 = f1_score(true_bond_types, pred_bond_types, average='weighted', zero_division=0)
        
        # Store metrics
        metrics['accuracy'].append(accuracy)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1'].append(f1)

    return metrics

def plot_metrics(accuracies, precisions, recalls, f1_scores, title='', outfile=''):
    values = [np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f1_scores)]
    stds = [np.std(accuracies), np.std(precisions), np.std(recalls), np.std(f1_scores)]
    plt.figure(figsize=(2.5,2))
    # Capitalize the first letter of each key
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    # face colors: white, gray, #2c939a, blue
    facecolors = ['white', 'gray', '#2c939a', 'black']
    # edge color: #000000, face color: 
    bars = plt.bar(labels, values, yerr=stds, edgecolor='#000000')
    for i, bar in enumerate(bars):
        bar.set_facecolor(facecolors[i])
    if title:
        plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()
    plt.clf()

def calculate_per_bond_type_metrics(edge_preds, edge_masks, edge_trues):
    metrics = {i: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} for i in range(len(bond_types))}

    # Process each molecule again for per-bond-type metrics
    for i in tqdm(range(len(edge_preds))):
        edge_pred = edge_preds[i]
        edge_mask = edge_masks[i]
        edge_true = edge_trues[i]
        
        valid_indices = torch.where(edge_mask.squeeze() == 1)[0]
        
        if len(valid_indices) == 0:
            continue
        
        pred_bond_types = torch.argmax(edge_pred.squeeze(), dim=1)[valid_indices].cpu().numpy()
        true_bond_types = torch.argmax(edge_true.squeeze(), dim=1)[valid_indices].cpu().numpy()
        
        # Calculate per-bond-type metrics
        for bond_type in range(len(bond_types)):
            # Create binary arrays for this bond type
            pred_binary = (pred_bond_types == bond_type).astype(int)
            true_binary = (true_bond_types == bond_type).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(true_binary, pred_binary)
            precision = precision_score(true_binary, pred_binary, zero_division=np.nan)
            recall = recall_score(true_binary, pred_binary, zero_division=np.nan)
            f1 = f1_score(true_binary, pred_binary, zero_division=np.nan)
            
            # Store metrics
            metrics[bond_type]['accuracy'].append(accuracy)
            if not np.isnan(precision):
                metrics[bond_type]['precision'].append(precision)
            if not np.isnan(recall):
                metrics[bond_type]['recall'].append(recall)
            if not np.isnan(f1):
                metrics[bond_type]['f1'].append(f1)

    return metrics

def plot_bond_type_metrics(json_data):
    for bond_type, metrics in json_data.items():
        if bond_type < 4:
            plot_metrics(metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1'], const.BOND_TYPE_NAMES[bond_type])

#%%
if __name__ == '__main__':
    # metrics = {name: calculate_metrics(*load_data(group)) for name, group in groups.items()}
    # for name, metric in metrics.items():
    #     print(name)
    # plot_metrics(metric['accuracy'], metric['precision'], metric['recall'], metric['f1'])
    
    print("\nPer-bond-type metrics:")
    bond_types = const.BOND_TYPE_NAMES
    metrics_per_bond_type = {name: calculate_per_bond_type_metrics(*load_data(group)) for name, group in groups.items()}
    for name, metric in metrics_per_bond_type.items():
        print(name)
        plot_bond_type_metrics(metric)
# %%

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
def plot_bond_probabilities(text_data, outname):
    # 解析文本数据
    categories = []
    probabilities = []
    for line in text_data.strip().split('\n'):
        name, value = line.split(':')
        categories.append(name)
        probabilities.append(float(value))
    
    # 创建横向条形图
    fig, ax = plt.subplots(figsize=(2, 1))
    
    # 基础颜色和透明度设置
    base_color = '#2c939a'
    max_alpha = 1.0
    min_alpha = 0.3  # 最小透明度
    
    # 计算每个条形的透明度（基于数值比例）
    alphas = min_alpha + (np.array(probabilities) / max(probabilities)) * (max_alpha - min_alpha)
    colors = [to_rgba(base_color, alpha=a) for a in alphas]
    
    # 绘制条形图（横向）
    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, probabilities, color=colors, 
                   edgecolor='black', linewidth=0.7, height=0.6)
    
    # 设置y轴
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, va='center')
    ax.tick_params(axis='y', length=0)
    
    # 移除x轴
    ax.xaxis.set_visible(False)
    
    # 移除上、右、下边框
    for spine in ['top', 'right', 'bottom']:
        ax.spines[spine].set_visible(False)
    
    # 保留左侧边框（y轴）
    ax.spines['left'].set_visible(True)
    
    # 在每个条形内部右侧标注数值
    for bar, prob in zip(bars, probabilities):
        width = bar.get_width()
        ax.text(prob+0.05, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.1f}%', 
                ha='left', 
                va='center', 
                color='black')
    
    # 调整布局
    ax.invert_yaxis()
    plt.xlim(0, max(probabilities) * 1.05)  # 留出少量空间
    plt.tight_layout()
    plt.savefig(outname)

text_data = ["""
Single:0.4774
Double:0.2494
Triple:0.2731
Aromatic:0.0001
""", """
Single:0.9034
Double:0.0436
Triple:0.0530
Aromatic:0.0000
""", """
Single:0.7761
Double:0.0187
Triple:0.2052
Aromatic:0.0000
""", """
Single:0.9737
Double:0.0260
Triple:0.0003
Aromatic:0.0000
""", """
Single:0.4826
Double:0.5173
Triple:0.0001
Aromatic:0.0000
""", """
Single:0.9820
Double:0.0177
Triple:0.0003
Aromatic:0.0000
""", """
Single:0.9197
Double:0.0115
Triple:0.0689
Aromatic:0.0000
""", """
Single:0.5504
Double:0.2501
Triple:0.1993
Aromatic:0.0001
"""]
if __name__ == '__main__':
    for i, text in enumerate(text_data):
        plot_bond_probabilities(text, f'analyses/figures/bond_probabilities_{i+1}.svg')
# %%
