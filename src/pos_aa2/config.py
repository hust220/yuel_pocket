# Configuration presets
CONFIGS = {
    'plinder': {
        'project': 'yuel_pocket',
        'exp_name': 'plinder_pos_aa2',
        'checkpoints': 'models',
        'logs': 'logs',
        'device': 'cuda',
        'log_iterations': 20,
        'wandb_entity': None,
        'enable_progress_bar': True,
        'lr': 1.0e-4, # Slightly reduced processing rate for stability with EGNN
        'batch_size': 8, # Better batch size for training
        'n_layers': 16, # Reasonable depth
        'n_epochs': 1000,
        'test_epochs': 20,
        'hidden_nf': 64, 
        'activation': 'silu',
        'resume': None,
        'seed': 42,
        'num_workers': 16,
        'contrastive_margin': 1,
        'sas_probe_radius': 1.4,
        'sas_n_points': 15,
    }
}

def get_config(mode='plinder'):
    if mode not in CONFIGS:
        raise ValueError(f"Unknown config mode: {mode}. Available modes: {list(CONFIGS.keys())}")
    return CONFIGS[mode].copy()
