import argparse
import os
import torch
import numpy as np
from rdkit import Chem
from scipy.spatial import KDTree
from .model import YuelPocket
from src.lightning import LightningWrapper
from .dataset import parse_protein, parse_molecule, build_graph
from src.graph import Graph, batch as graph_batch
from src.pdb_utils import Structure, get_sas_points_shrake_rupley, pdb_line
from src.utils import pick_latest
from .config import get_config
from src import const
from io import StringIO
import warnings

def save_pdb_with_bfactor(coords, scores, pdb_out, atom_name=' H  ', res_name='PRO'):
    with open(pdb_out, 'w') as f:
        for i, (coord, score) in enumerate(zip(coords, scores)):
            line = pdb_line(record="HETATM",
                            atom_id=i+1,
                            atom_name=atom_name,
                            alt_loc=" ",
                            res_name=res_name,
                            chain_id="A",
                            res_id=i+1,
                            insertion=" ",
                            x=coord[0],
                            y=coord[1],
                            z=coord[2],
                            occupancy=1.0,
                            temp_factor=score,
                            element="H",
                            charge="  ")
            f.write(line + "\n")

def run_residues_inference(pdb_path, model_patterns, output_txt, save_pdb=False, device='cpu'):
    model_path = pick_latest(model_patterns)
    print(f"Loading model from {model_path}...")
    
    try:
        model = LightningWrapper.load_from_checkpoint(
            model_path,
            map_location=device,
            model_class=YuelPocket,
            weights_only=False,
            strict=False
        )
    except Exception as e:
        print(f"Error loading checkpoint with LightningWrapper: {e}")
        return

    model.to(device)
    model.eval()

    # Parse Data
    print("Parsing PDB...")
    with open(pdb_path, 'r') as f:
        pdb_content = f.read()
    
    prot_pos, prot_h, prot_contacts, _ = parse_protein(pdb_content)
    # No ligand parsing needed for inference
    
    if len(prot_pos) == 0:
        print("No protein residues found.")
        return
    
    # Calculate SAS points (Probes)
    print("Calculating SAS points...")
    structure = Structure()
    try:
        structure.read(StringIO(pdb_content))
        sas_points, _ = get_sas_points_shrake_rupley(structure, probe_radius=1.4, n_points_per_atom=15, target_points=None)
    except Exception as e:
        print(f"SAS calculation error: {e}")
        return
        
    if len(sas_points) == 0:
        print("No SAS points found.")
        return

    print(f"Building graph with {len(sas_points)} probes...")
    
    # Use generic build_graph from dataset.py
    # Pass empty arrays for ligand data as it is not used in inference for this model type (wo_lig)
    try:
        g = build_graph(
            protein_name="inference",
            ligand_name="inference",
            mol_pos=np.empty((0, 3)),
            mol_h=None, 
            mol_bonds=None, 
            prot_pos=prot_pos,
            prot_h=prot_h,
            prot_contacts=prot_contacts,
            sas_points=sas_points, 
            pick_samples=False # INFERENCE MODE: Use all SAS points
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Graph build raised exception: {e}")
        return
    
    if g is None:
        print("Failed to build graph (returned None).")
        return

    # Move to device
    g.to(device)
    
    # Run Inference
    print("Running inference...")
    with torch.no_grad():
        logits = model.sample_chain(g=g)
    
    # Extract Probe Scores
    # The last len(sas_points) nodes are the probes
    n_probes = len(sas_points)
    probe_scores = logits[-n_probes:].cpu().numpy()
    
    # Save Output
    print(f"Saving results to {output_txt}...")
    with open(output_txt, 'w') as f:
        f.write("X,Y,Z,Score\n")
        for p, s in zip(sas_points, probe_scores):
            f.write(f"{p[0]:.4f},{p[1]:.4f},{p[2]:.4f},{s:.4f}\n")
            
    if save_pdb:
        pdb_out = output_txt.replace('.txt', '_sas.pdb')
        print(f"Saving SAS PDB to {pdb_out}...")
        save_pdb_with_bfactor(sas_points, probe_scores, pdb_out, res_name='SAS')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb', help='Path to protein PDB')
    # parser.add_argument('ligand', help='Path to ligand (SDF or MOL)') # Removed
    parser.add_argument('output', help='Path to output TXT')
    parser.add_argument('--model', default=None, required=False, help='Path to model checkpoint (optional)')
    parser.add_argument('--save_pdb', action='store_true', help='Save SAS points PDB')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    
    args = parser.parse_args()

    config = get_config()
    base_dir = config.get('checkpoints')
    if args.model is None:
        target = config.get('exp_name')
    else:
        target = args.model
        
    model_patterns = [
        target if target.endswith('.ckpt') else os.path.join(target, '**/*.ckpt'),
        os.path.join(base_dir, f"{target}*", "**/*.ckpt")
    ]
    
    run_residues_inference(args.pdb, model_patterns, args.output, args.save_pdb, args.device)
