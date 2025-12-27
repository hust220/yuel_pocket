import argparse
import os
import sys
import numpy as np

# Ensure the script's directory is in sys.path for relative imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Default model path relative to the script location
DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'models', 'pocketminer'))

# Now imports can work regardless of CWD
import mdtraj as md
import tensorflow as tf
from models import MQAModel
from util import load_checkpoint
from validate_performance_on_xtals import process_strucs

def predict_step(model, X, S, mask):
    """Run inference without reloading checkpoint."""
    prediction = model(X, S, mask, train=False, res_level=True)
    return prediction

def predict_single(model, pdb_path, output_path):
    """Run prediction for a single PDB using a pre-loaded model."""
    if not os.path.exists(pdb_path):
        print(f"Error: PDB file {pdb_path} not found.")
        return
    
    try:
        t = md.load(pdb_path)
        # process_strucs expects a list of trajectories
        X, S, mask = process_strucs([t])
        
        # 3. Inference
        predictions = predict_step(model, X, S, mask)
        scores = predictions[0]
        
        # Slice scores to the actual protein backbone length
        prot_iis = t.top.select("protein and (name N or name CA or name C or name O)")
        prot_bb = t.atom_slice(prot_iis)
        n_res = prot_bb.top.n_residues
        
        final_scores = scores[:n_res]
        
        # 4. Save Output
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        if output_path.endswith('.pdb'):
            res_map = {}
            for i, res in enumerate(prot_bb.top.residues):
                res_map[(res.chain.chain_id, res.resSeq)] = final_scores[i]
                
            with open(pdb_path, 'r') as f_in, open(output_path, 'w') as f_out:
                for line in f_in:
                    if line.startswith("ATOM"):
                        try:
                            # Use 1-based indexing for residue sequence as per PDB format
                            # and matching the key used in res_map
                            chain_id = line[21]
                            res_seq = int(line[22:26].strip())
                            key = (chain_id, res_seq)
                            if key in res_map:
                                score = res_map[key]
                                line = line[:60] + f"{score:6.2f}" + line[66:]
                            else:
                                # Default b-factor if not in map (e.g. non-backbone atoms if desired, 
                                # but usually all atoms of residue should have same score)
                                line = line[:60] + f"{0.00:6.2f}" + line[66:]
                        except:
                            pass
                    f_out.write(line)
            print(f"Saved annotated PDB to {output_path}")
        else:
            # Default as .txt
            np.savetxt(output_path, final_scores, fmt='%.4g', delimiter='\n')
            print(f"Saved predictions to {output_path}")
    except Exception as e:
        print(f"Error predicting for {pdb_path}: {e}")

def run_prediction(tasks, model_path, dropout=0.1, n_layers=4, hidden_dim=100):
    # 1. Load Model
    print(f"Initializing model...")
    model = MQAModel(node_features=(8, 50), edge_features=(1, 32),
                     hidden_dim=(16, hidden_dim),
                     num_layers=n_layers, dropout=dropout)
    
    # Load weights once
    print(f"Loading weights from {model_path}...")
    # Using default optimizer as in validate_performance_on_xtals
    opt = tf.keras.optimizers.Adam()
    load_checkpoint(model, opt, model_path)
    
    # 2. Iterate tasks
    for i, (pdb_path, output_path) in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] Predicting for {os.path.basename(pdb_path)}...")
        predict_single(model, pdb_path, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PocketMiner Prediction')
    parser.add_argument('pdb', nargs='?', help='Path to input PDB file (if not using --list)')
    parser.add_argument('output', nargs='?', help='Path to output file (if not using --list)')
    parser.add_argument('--list', help='Path to task list file (space-separated input output)')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH, help=f'Path to model checkpoint (Default: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=100)
    
    args = parser.parse_args()
    
    tasks = []
    if args.list:
        if not os.path.exists(args.list):
            print(f"Error: Task list {args.list} not found.")
            sys.exit(1)
        with open(args.list, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    tasks.append((parts[0], parts[1]))
    elif args.pdb and args.output:
        tasks.append((args.pdb, args.output))
    else:
        print("Error: Must provide either 'pdb output' or '--list path/to/list'")
        parser.print_help()
        sys.exit(1)
        
    run_prediction(
        tasks, 
        args.model, 
        dropout=args.dropout, 
        n_layers=args.n_layers, 
        hidden_dim=args.hidden_dim
    )
