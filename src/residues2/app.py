import argparse
import os
import torch
import numpy as np
from rdkit import Chem
from scipy.spatial import KDTree
from .model import YuelPocket
from src.lightning import LightningWrapper
from .dataset import parse_protein, parse_molecule, build_graph, Featurizer
from src.graph import Graph, batch as graph_batch
from src.pdb_utils import Structure, pdb_line, get_sas_points_shrake_rupley
from src.utils import pick_latest, disable_rdkit_logging
from .config import get_config
from src import const
from src.clustering import hill_climbing_cluster
from io import StringIO
import warnings

def save_sas_pdb(sas_points, sas_scores, output_path):
    print(f"Saving SAS points to {output_path}...")
    with open(output_path, 'w') as f:
        for i, (coord, score) in enumerate(zip(sas_points, sas_scores)):
            line = pdb_line(record="HETATM",
                             atom_id=i+1,
                             atom_name=" H  ",
                             alt_loc=" ",
                             res_name="SAS",
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

def save_clusters_pdb(clusters_info, sas_points, sas_scores, output_path):
    print(f"Saving clusters to {output_path}...")
    with open(output_path, 'w') as f:
        atom_count = 1
        for cluster in clusters_info:
            cid = cluster['id'] + 1
            for idx in cluster['indices']:
                coord = sas_points[idx]
                score = sas_scores[idx]
                line = pdb_line(record="ATOM",
                                atom_id=atom_count,
                                atom_name=' H  ',
                                alt_loc=" ",
                                res_name='PKT',
                                chain_id="A",
                                res_id=cid,
                                insertion=" ",
                                x=coord[0],
                                y=coord[1],
                                z=coord[2],
                                occupancy=1.0,
                                temp_factor=score,
                                element="H",
                                charge="  ")
                f.write(line + "\n")
                atom_count += 1

def run_residues_inference(model, pdb_path, ligand_path, output_path, device='cpu', do_cluster=False, k_nn=10):
    # Parse Data
    print(f"Processing {pdb_path} with {ligand_path}...")
    if not os.path.exists(pdb_path):
        print(f"Error: PDB file {pdb_path} not found.")
        return
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
        
    # Re-implementing parse_protein logic to track residue metadata
    structure = Structure()
    structure.read(StringIO(pdb_content), skip_hetatm=True)
    
    res_metadata = [] # List of (chain_id, res_id, res_name)
    prot_pos = []
    prot_one_hot = []
    
    backbone_atoms = {'N', 'CA', 'C', 'O', 'H', 'HA', 'HA2', 'HA3', 'OXT'}
    res_sidechains = [] # List of sidechain coords for each unique residue
    
    for model_struct in structure:
        for chain in model_struct:
            sorted_residues = sorted(chain.residues, key=lambda r: r.res_id)
            for residue in sorted_residues:
                if residue.res_name in const.ALLOWED_RESIDUE_TYPES:
                    ca = residue.get_atom('CA')
                    if ca:
                        res_info = (chain.chain_id, residue.res_id, residue.res_name)
                        # 1. BB (CA) Node
                        ca_coord = ca.get_coord()
                        prot_pos.append(ca_coord)
                        prot_one_hot.append(Featurizer.aa_one_hot('BB'))
                        res_metadata.append(res_info)
                        
                        # 2. SC Center Node
                        sc_atoms = [a for a in residue.atoms if a.atom_name not in backbone_atoms]
                        if len(sc_atoms) > 0:
                            sc_coords = np.array([a.get_coord() for a in sc_atoms])
                            sc_center = np.mean(sc_coords, axis=0)
                        else:
                            sc_center = ca_coord
                        prot_pos.append(sc_center)
                        prot_one_hot.append(Featurizer.aa_one_hot(residue.res_name))
                        res_metadata.append(res_info)
                        
                        # Store sidechain atoms for SAS scoring
                        sc_atoms_raw = [a.get_coord() for a in residue.atoms if a.atom_name not in backbone_atoms]
                        if not sc_atoms_raw:
                            sc_atoms_raw = [ca_coord]
                        res_sidechains.append(np.array(sc_atoms_raw))
        break 
        
    if not prot_pos:
        print("No protein residues found.")
        return
        
    prot_pos = np.array(prot_pos)
    prot_one_hot = np.array(prot_one_hot)
    
    # Compute Contacts (8.0A)
    tree = KDTree(prot_pos)
    pairs = tree.query_pairs(r=8.0)
    prot_contacts = np.array([[i+1, j+1, np.linalg.norm(prot_pos[i]-prot_pos[j])] for i, j in pairs]) if pairs else np.zeros((0, 3))

    mol_pos, mol_h, mol_bonds = parse_molecule(mol)
    
    # Build Graph
    g = build_graph("inference", "inference", mol_pos, mol_h, mol_bonds, prot_pos, prot_one_hot, prot_contacts)
    if g is None:
        print("Failed to build graph.")
        return
    
    # Manual batch reset: build_graph returns a batched graph (prot + ligs). 
    # We need to treat them as ONE complex for the model loop.
    g.batch_size = 1
    g.batch_num_nodes = [g.num_nodes]
    g.batch_num_edges = [g.num_edges]
    
    g.to(device)
    
    # Run Inference
    with torch.no_grad():
        pair_probs, pock_probs = model.sample_chain(g=g)
    
    if not pair_probs or not pock_probs:
        print("Inference returned no results.")
        return
        
    # Extract Results
    pairing_prob = pair_probs[0][0].item()
    pocket_probs = pock_probs[0].cpu().numpy() # Length 2N (BB and SC for each residue)
    
    # Consolidate residue scores (Max of BB and SC)
    unique_residues = []
    res_scores = []
    
    current_res = None
    scores_for_res = []
    
    for i, meta in enumerate(res_metadata):
        if meta != current_res:
            if scores_for_res:
                unique_residues.append(current_res)
                res_scores.append(max(scores_for_res))
            current_res = meta
            scores_for_res = [pocket_probs[i]]
        else:
            scores_for_res.append(pocket_probs[i])
            
    if scores_for_res:
        unique_residues.append(current_res)
        res_scores.append(max(scores_for_res))
    
    print(f"Overall Binding Probability: {pairing_prob:.4f}")
    
    # Save Output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    base_output, _ = os.path.splitext(output_path)
    
    # Save Residue-level Output (CSV/TXT) - Always generated
    txt_output = base_output + '.txt'
    with open(txt_output, 'w') as f:
        f.write(f"# Overall Binding Probability: {pairing_prob:.6f}\n")
        f.write("Chain,ResID,ResName,PocketProbability\n")
        for (ch, rid, rname), score in zip(unique_residues, res_scores):
            f.write(f"{ch},{rid},{rname},{score:.6f}\n")
    print(f"Residue scores saved to {txt_output}")

    # Save Annotated PDB if requested
    if output_path.endswith('.pdb'):
        print(f"Saving annotated PDB to {output_path}...")
        res_map = {(ch, int(rid)): score for (ch, rid, rname), score in zip(unique_residues, res_scores)}
        with open(pdb_path, 'r') as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                if line.startswith("ATOM"):
                    try:
                        chain_id = line[21]
                        res_id = int(line[22:26].strip())
                        if (chain_id, res_id) in res_map:
                            score = res_map[(chain_id, res_id)]
                            line = line[:60] + f"{score:6.2f}" + line[66:]
                        else:
                            line = line[:60] + f"{0.00:6.2f}" + line[66:]
                        f_out.write(line)
                    except:
                        pass

    # --- SAS and Clustering logic ---
    print("Generating SAS points...")
    try:
        sas_points, _ = get_sas_points_shrake_rupley(structure, probe_radius=1.4, n_points_per_atom=15)
    except Exception as e:
        print(f"SAS calculation failed: {e}")
        return

    if len(sas_points) == 0:
        print("No SAS points generated.")
        return

    # Score SAS points: Sum of scores of residues within 6A of sidechain atoms
    # Each residue's score is added only once per SAS point
    print("Scoring SAS points...")
    sas_scores = np.zeros(len(sas_points))
    
    # Build a KDTree for each residue sidechain? No, that's too slow.
    # Build one KDTree for all sidechain atoms and track residue index.
    all_sc_coords = []
    sc_res_indices = []
    for i, sc in enumerate(res_sidechains):
        all_sc_coords.extend(sc)
        sc_res_indices.extend([i] * len(sc))
    
    all_sc_coords = np.array(all_sc_coords)
    sc_tree = KDTree(all_sc_coords)
    
    # For each SAS point, find all sidechain atoms within 6A
    neighbor_indices = sc_tree.query_ball_point(sas_points, r=6.0)
    
    for i, neighbors in enumerate(neighbor_indices):
        if not neighbors:
            continue
        # Get unique residues among these neighbors
        unique_res_indices = set(sc_res_indices[n] for n in neighbors)
        sas_scores[i] = sum(res_scores[ri] for ri in unique_res_indices)

    base_output, _ = os.path.splitext(output_path)
    # Always save SAS PDB
    sas_pdb = base_output + '_sas.pdb'
    save_sas_pdb(sas_points, sas_scores, sas_pdb)

    if do_cluster:
            print("Running clustering...")
            cluster_labels, clusters_info = hill_climbing_cluster(sas_points, sas_scores, k_nn=k_nn)
            print(f"Found {len(clusters_info)} clusters.")
            
            cluster_csv = base_output + '_clusters.csv'
            with open(cluster_csv, 'w') as f:
                f.write("ClusterID,X,Y,Z,Score\n")
                for cluster in clusters_info:
                    cid = cluster['id']
                    for idx in cluster['indices']:
                        p = sas_points[idx]
                        s = sas_scores[idx]
                        f.write(f"{cid},{p[0]:.4f},{p[1]:.4f},{p[2]:.4f},{s:.4f}\n")
            print(f"Clusters saved to {cluster_csv}")

            cluster_pdb = base_output + '_clusters.pdb'
            save_clusters_pdb(clusters_info, sas_points, sas_scores, cluster_pdb)

def load_model(model_patterns, device='cpu'):
    model_path = pick_latest(model_patterns)
    if not model_path:
        print(f"No checkpoint found matching {model_patterns}")
        return None
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
        print(f"Error loading checkpoint: {e}")
        return None

    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YuelPocket Residues Inference')
    parser.add_argument('pdb', nargs='?', help='Path to protein PDB')
    parser.add_argument('ligand', nargs='?', help='Path to ligand (SDF/MOL/PDB)')
    parser.add_argument('output', nargs='?', help='Path to output TXT')
    parser.add_argument('--list', help='File containing list of (pdb ligand output) per line')
    parser.add_argument('--model', default=None, help='Path to model checkpoint or experiment name')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--cluster', action='store_true', help='Enable clustering of SAS points')
    parser.add_argument('--k_nn', type=int, default=30, help='Nearest neighbors for clustering')
    
    args = parser.parse_args()
    disable_rdkit_logging()

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
    # Resolve absolute path to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_dir = os.path.join(project_root, config.get('checkpoints', 'models'))
    
    if args.model is None:
        target = config.get('exp_name') + '_bs'
    else:
        target = args.model
        
    model_patterns = [
        target if target.endswith('.ckpt') else os.path.join(target, '**/*.ckpt'),
        os.path.join(base_dir, f"{target}*", "**/*.ckpt"),
        os.path.join(base_dir, target, "**/*.ckpt")
    ]
    
    model = load_model(model_patterns, args.device)
    if model is None:
        exit(1)

    for pdb, ligand, output in tasks:
        try:
            run_residues_inference(model, pdb, ligand, output, args.device, 
                                   do_cluster=args.cluster, k_nn=args.k_nn)
        except Exception as e:
            print(f"Failed to process {pdb}: {e}")
            import traceback
            traceback.print_exc()
