import argparse
import os
import torch
import numpy as np
from rdkit import Chem
from scipy.spatial import KDTree
from src.residues.model import YuelPocket
from src.lightning import LightningWrapper
from src.residues.dataset import parse_protein, parse_molecule, build_graph, get_sas_points_shrake_rupley, PocketDataset
from src.graph import Graph, batch as graph_batch
from src.pdb_utils import Structure
from src.utils import pick_latest
from src.residues.config import get_config
from src import const
from io import StringIO
import warnings
from src.pdb_utils import pdb_line

def save_pdb_with_bfactor(coords, scores, pdb_out, atom_name=' CA ', res_name='PRO'):
    with open(pdb_out, 'w') as f:
        for i, (coord, score) in enumerate(zip(coords, scores)):
            line = pdb_line(record="ATOM",
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
                            element="C",
                            charge="  ")
            f.write(line + "\n")

def run_residues_inference(pdb_path, ligand_path, model_patterns, output_txt, save_pdb=False, save_prot_pdb=False, device='cpu'):
    model_path = pick_latest(model_patterns)
    print(f"Loading model from {model_path}...")
    
    try:
        # Load model using LightningWrapper
        # Note: map_location handles device placement for loading, but we explicit .to(device) later
        model = LightningWrapper.load_from_checkpoint(
            model_path,
            map_location=device,
            model_class=YuelPocket,
            dataset_class=PocketDataset,
            weights_only=False,
            strict=False # Allow loose matching if exact architecture varies slightly
        )
    except Exception as e:
        print(f"Error loading checkpoint with LightningWrapper: {e}")
        return

    model.to(device)
    model.eval()

    # Parse Data
    print("Parsing PDB and Ligand...")
    with open(pdb_path, 'r') as f:
        pdb_content = f.read()
    
    if not os.path.exists(ligand_path):
        print(f"Error: Ligand file not found at {ligand_path}")
        return
    if os.path.getsize(ligand_path) == 0:
        print(f"Error: Ligand file {ligand_path} is empty.")
        return

    if ligand_path.endswith('.mol'):
        mol = Chem.MolFromMolFile(ligand_path, sanitize=False)
    else:
        # Assume SDF
        mol = Chem.SDMolSupplier(ligand_path, sanitize=False)[0]

    if mol is None:
        print("Error parsing Ligand.")
        return
        
    prot_pos, prot_h, prot_contacts, prot_backbone, _ = parse_protein(pdb_content)
    mol_pos, mol_h, mol_bonds = parse_molecule(mol)
    
    if len(prot_pos) == 0:
        print("No protein residues found.")
        return
        
    # Build Graph using dataset utility
    g = build_graph(
        protein_name="inference",
        ligand_name="inference",
        mol_pos=mol_pos,
        mol_h=mol_h,
        mol_bonds=mol_bonds,
        prot_pos=prot_pos,
        prot_h=prot_h,
        prot_contacts=prot_contacts,
        prot_backbone=prot_backbone,
        sas_points=None # Not needed for inference graph structure
    )
    
    if g is None:
        print("Failed to build graph (protein might be too large).")
        return

    # Move graph to device
    for k in g.ndata:
        g.ndata[k] = g.ndata[k].to(device)
    for k in g.edata:
        g.edata[k] = g.edata[k].to(device)
    g.edge_index = g.edge_index.to(device)
    
    # Run Inference
    print("Running inference...")
    with torch.no_grad():
        logits = model.sample_chain(g=g)
    
    # Extract Protein Scores (first p_size nodes)
    p_size = len(prot_pos)
    prot_scores = logits[:p_size].cpu().numpy()
    
    # Calculate SAS SAS
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
        
    print(f"Assigning scores to {len(sas_points)} SAS points...")
    tree = KDTree(prot_pos)
    
    radius = 10.0
    queries = tree.query_ball_point(sas_points, r=radius)
    
    sas_scores = []
    for indices in queries:
        if len(indices) == 0:
            sas_scores.append(0.0)
        else:
            mean_score = np.mean(prot_scores[indices])
            sas_scores.append(mean_score)
            
    # Save Output
    print(f"Saving results to {output_txt}...")
    with open(output_txt, 'w') as f:
        f.write("X,Y,Z,Score\n")
        for p, s in zip(sas_points, sas_scores):
            f.write(f"{p[0]:.4f},{p[1]:.4f},{p[2]:.4f},{s:.4f}\n")
            
    if save_pdb:
        pdb_out = output_txt.replace('.txt', '_sas.pdb')
        print(f"Saving SAS PDB to {pdb_out}...")
        save_pdb_with_bfactor(sas_points, sas_scores, pdb_out, res_name='SAS')
        
    if save_prot_pdb:
        pdb_prot_out = output_txt.replace('.txt', '_prot.pdb')
        print(f"Saving Protein PDB with scores to {pdb_prot_out}...")
        
        # Parse structure to map scores back to all atoms
        try:
            full_structure = Structure()
            full_structure.read(StringIO(pdb_content))
            
            score_idx = 0
            # Match logic from parse_protein in dataset.py
            for model in full_structure.models:
                for chain in model.chains:
                    for residue in chain.residues:
                        # Check if this residue was included in the graph
                        if residue.res_name in const.ALLOWED_RESIDUE_TYPES and residue.get_atom('CA'):
                            if score_idx < len(prot_scores):
                                score = float(prot_scores[score_idx])
                                for atom in residue.atoms:
                                    atom.temp_factor = score
                                score_idx += 1
                        else:
                            # Not in graph, zero out b-factor
                            for atom in residue.atoms:
                                atom.temp_factor = 0.0
                break # dataset.py only converts the first model
            
            full_structure.write(pdb_prot_out)
            
        except Exception as e:
            print(f"Error saving protein PDB: {e}")
            # Fallback to CA-only if full save fails
            save_pdb_with_bfactor(prot_pos, prot_scores, pdb_prot_out, atom_name=' CA ', res_name='PRO')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', required=True, help='Path to protein PDB')
    parser.add_argument('--ligand', required=True, help='Path to ligand (SDF or MOL)')
    parser.add_argument('--model', default=None, required=False, help='Path to model checkpoint (optional)')
    parser.add_argument('--output', required=True, help='Path to output TXT')
    parser.add_argument('--save_pdb', action='store_true', help='Save SAS points PDB')
    parser.add_argument('--save_prot_pdb', action='store_true', help='Save Protein PDB with scores')
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
    
    run_residues_inference(args.pdb, args.ligand, model_patterns, args.output, args.save_pdb, args.save_prot_pdb, args.device)
