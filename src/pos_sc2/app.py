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
from src.clustering import hill_climbing_cluster
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

def save_clusters_pdb(clusters_info, sas_points, probe_scores, pdb_out):
    with open(pdb_out, 'w') as f:
        atom_count = 1
        for cluster in clusters_info:
            cid = cluster['id'] + 1 # 1-based cluster ID for PDB residue number
            for idx in cluster['indices']:
                coord = sas_points[idx]
                score = probe_scores[idx]
                line = pdb_line(record="ATOM",
                                atom_id=atom_count,
                                atom_name=' C  ', # Use C for point
                                alt_loc=" ",
                                res_name='PKT', # Pocket
                                chain_id="A",
                                res_id=cid,     # Cluster ID as Residue ID
                                insertion=" ",
                                x=coord[0],
                                y=coord[1],
                                z=coord[2],
                                occupancy=1.0,
                                temp_factor=score,
                                element="C",
                                charge="  ")
                f.write(line + "\n")
                atom_count += 1

def load_model(model_patterns, device='cpu'):
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
        return None

    model.to(device)
    model.eval()
    return model

def run_residues_inference(model, pdb_path, ligand_path, output_txt, save_pdb=False, device='cpu', do_cluster=False, k_nn=10):

    # Parse Data
    print("Parsing PDB and Ligand...")
    with open(pdb_path, 'r') as f:
        pdb_content = f.read()
    
    if not os.path.exists(ligand_path) or os.path.getsize(ligand_path) == 0:
        print(f"Error: Invalid ligand file {ligand_path}")
        return

    if ligand_path.endswith('.mol'):
        mol = Chem.MolFromMolFile(ligand_path, sanitize=False)
    elif ligand_path.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(ligand_path, sanitize=False)
    else:
        mol = Chem.SDMolSupplier(ligand_path, sanitize=False)[0]

    if mol is None:
        print("Error parsing Ligand.")
        return
        
    prot_pos, prot_h, prot_contacts, _ = parse_protein(pdb_content)
    mol_pos, mol_h, mol_bonds = parse_molecule(mol)
    
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
    try:
        g = build_graph(
            protein_name="inference",
            ligand_name="inference",
            mol_pos=mol_pos,
            mol_h=mol_h,
            mol_bonds=mol_bonds,
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
    n_probes = len(sas_points)
    probe_scores = logits[-n_probes:].cpu().numpy()
    
    # Clustering (Optional)
    if do_cluster:
        print("Running clustering...")
        cluster_labels, clusters_info = hill_climbing_cluster(sas_points, probe_scores, k_nn=k_nn)
        print(f"Found {len(clusters_info)} clusters.")
        
        # Save Clusters CSV
        cluster_out = output_txt.replace('.txt', '_clusters.csv')
        print(f"Saving clusters to {cluster_out}...")
        with open(cluster_out, 'w') as f:
            f.write("ClusterID,X,Y,Z,Score\n")
            for cluster in clusters_info:
                cid = cluster['id']
                for idx in cluster['indices']:
                    pct = sas_points[idx]
                    sco = probe_scores[idx] 
                    f.write(f"{cid},{pct[0]:.4f},{pct[1]:.4f},{pct[2]:.4f},{sco:.4f}\n")

        # Save Clusters PDB
        if save_pdb:
            cluster_pdb_out = output_txt.replace('.txt', '_clusters.pdb')
            print(f"Saving clusters PDB to {cluster_pdb_out}...")
            save_clusters_pdb(clusters_info, sas_points, probe_scores, cluster_pdb_out)
    
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
    parser.add_argument('pdb', nargs='?', help='Path to protein PDB')
    parser.add_argument('ligand', nargs='?', help='Path to ligand (SDF or MOL)')
    parser.add_argument('output', nargs='?', help='Path to output TXT')
    parser.add_argument('--list', help='File containing list of (pdb ligand output) per line')
    parser.add_argument('--model', default=None, required=False, help='Path to model checkpoint (optional)')
    parser.add_argument('--save_pdb', action='store_true', help='Save SAS points PDB')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--cluster', action='store_true', help='Enable clustering of predictions (Hill-Climbing style)')
    parser.add_argument('--k_nn', type=int, default=10, help='Number of nearest neighbors for hill climbing')
    
    args = parser.parse_args()

    # Determine tasks
    tasks = []
    if args.list:
        if not os.path.exists(args.list):
            print(f"List file not found: {args.list}")
            exit(1)
        with open(args.list, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    tasks.append((parts[0], parts[1], parts[2]))
                elif len(parts) > 0:
                    print(f"Warning: line '{line.strip()}' does not have 3 components, skipping.")
    elif args.pdb and args.ligand and args.output:
        tasks.append((args.pdb, args.ligand, args.output))
    else:
        parser.print_help()
        print("\nError: Either provide (pdb, ligand, output) or --list")
        exit(1)

    config = get_config()
    base_dir = config.get('checkpoints')
    if args.model is None:
        target = config.get('exp_name') + '_bs'
    else:
        target = args.model
        
    model_patterns = [
        target if target.endswith('.ckpt') else os.path.join(target, '**/*.ckpt'),
        os.path.join(base_dir, f"{target}*", "**/*.ckpt")
    ]
    
    model = load_model(model_patterns, args.device)
    if model is None:
        exit(1)

    for pdb, ligand, output in tasks:
        print(f"\nProcessing {pdb}...")
        try:
            run_residues_inference(model, pdb, ligand, output, args.save_pdb, args.device, args.cluster, args.k_nn)
        except Exception as e:
            print(f"Failed to process {pdb}: {e}")
            import traceback
            traceback.print_exc()
