import argparse
import os
import torch
import numpy as np
from rdkit import Chem
from src.positions2.model import YuelPocket
from src.positions2.dataset import parse_protein, parse_molecule, build_sample_features, PocketDataset
from src.lightning import LightningWrapper
from src.pdb_utils import get_sas_points_shrake_rupley, pdb_line, Structure
from src.positions2.config import get_config
from src.utils import pick_latest
from io import StringIO
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

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


def run_inference(pdb_path, ligand_path, model_patterns, output_txt, save_pdb=False, device='cpu'):
    # Find model checkpoint
    try:
        model_path = pick_latest(model_patterns)
    except FileNotFoundError:
        print(f"No checkpoints found matching patterns: {model_patterns}")
        return
            
    print(f"Loading model from {model_path}...")
    
    # Load Config to get normalization_factor
    config = get_config()
    norm_factor = config.get('normalization_factor', 1000.0)
    
    # Load Model
    try:
        model = LightningWrapper.load_from_checkpoint(
            model_path,
            map_location=device,
            model_class=YuelPocket,
            dataset_class=PocketDataset,
            strict=False,
            weights_only=False
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
    
    if ligand_path.endswith('.mol'):
        mol = Chem.MolFromMolFile(ligand_path, sanitize=False)
    else:
        mol = Chem.SDMolSupplier(ligand_path, sanitize=False)[0]

    if mol is None:
        print("Error parsing Ligand.")
        return
        
    prot_pos, prot_h, prot_contacts, prot_backbone = parse_protein(pdb_content)
    mol_pos, mol_h, mol_bonds = parse_molecule(mol)
    
    structure = Structure()
    structure.read(StringIO(pdb_content))
    
    print("Generating SAS points...")
    # Calculate ALL SAS points for inference
    sas_points, _ = get_sas_points_shrake_rupley(structure, probe_radius=1.4, n_points_per_atom=15, target_points=None)
    
    if len(sas_points) == 0:
        print("No SAS points found.")
        return

    # Build Features
    # Note: build_sample_features might expect arguments
    # signature: (protein_name, ligand_name, mol_pos, mol_h, mol_bonds, prot_pos, prot_h, prot_contacts, prot_backbone, sas_points=None, normalization_factor=1000.0)
    
    try:
        sample = build_sample_features(
            protein_name="inference",
            ligand_name="inference",
            mol_pos=mol_pos,
            mol_h=mol_h,
            mol_bonds=mol_bonds,
            prot_pos=prot_pos,
            prot_h=prot_h,
            prot_contacts=prot_contacts,
            prot_backbone=prot_backbone,
            sas_points=sas_points,
            normalization_factor=norm_factor
        )
    except Exception as e:
        print(f"Error building features: {e}")
        return

    if sample is None:
        print("Failed to build sample (protein might be too large).")
        return
        
    # Construct Batch
    # prot: [Np, 46], mol: [Nm, 17], sas: [Ns, 6]
    # Need to add Batch dimension [1, ...]
    
    batch = {}
    batch['prot'] = torch.tensor(sample['prot'], dtype=torch.float).unsqueeze(0).to(device)
    batch['mol'] = torch.tensor(sample['mol'], dtype=torch.float).unsqueeze(0).to(device)
    batch['sas'] = torch.tensor(sample['sas'], dtype=torch.float).unsqueeze(0).to(device)
    
    # Masks
    p_len = sample['prot'].shape[0]
    m_len = sample['mol'].shape[0]
    max_pm = p_len + m_len
    
    prot_mask = torch.zeros((1, max_pm), dtype=torch.bool)
    prot_mask[0, :p_len] = True
    
    mol_mask = torch.zeros((1, max_pm), dtype=torch.bool)
    mol_mask[0, p_len:] = True
    
    batch['prot_mask'] = prot_mask.to(device)
    batch['mol_mask'] = mol_mask.to(device)
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        sas_scores = model.sample_chain(batch=batch) 
        # Returns [1, Ns]
        
    scores = sas_scores[0].cpu().numpy()
    
    # Save Output
    print(f"Saving results to {output_txt}...")
    with open(output_txt, 'w') as f:
        f.write("X,Y,Z,Score\n")
        for p, s in zip(sas_points, scores):
            f.write(f"{p[0]:.4f},{p[1]:.4f},{p[2]:.4f},{s:.4f}\n")
            
    if save_pdb:
        pdb_out = output_txt.replace('.txt', '_sas.pdb')
        print(f"Saving SAS PDB to {pdb_out}...")
        save_pdb_with_bfactor(sas_points, scores, pdb_out, res_name='SAS')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb', help='Path to protein PDB')
    parser.add_argument('ligand', help='Path to ligand (SDF or MOL)')
    parser.add_argument('output', help='Path to output TXT')
    parser.add_argument('--model', default='plinder_positions_bs8', required=False, help='Path to model checkpoint (optional)')
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
    
    run_inference(args.pdb, args.ligand, model_patterns, args.output, args.save_pdb, args.device)
