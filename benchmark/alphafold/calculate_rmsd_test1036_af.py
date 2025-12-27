
import os
import warnings
from Bio.PDB import PDBParser, Superimposer
import pandas as pd
from multiprocessing import Pool, cpu_count

# Suppress Bio.PDB warnings
warnings.filterwarnings("ignore")

af_dir = "/home/tyq4zn/scratch/codes/yuel_pocket/benchmark/alphafold/test1036_af"
ground_truth_dir = "/home/tyq4zn/scratch/codes/yuel_pocket/benchmark/alphafold/test1036"
output_csv = "/home/tyq4zn/scratch/codes/yuel_pocket/benchmark/alphafold/rmsd_test1036_af.csv"

def get_ca_atoms(structure):
    atoms = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    res_id = residue.id
                    key = (chain.id, res_id)
                    atoms[key] = residue['CA']
        break # Only first model
    return atoms

def process_system(filename):
    system_id = filename.replace("_protein.pdb", "")
    af_path = os.path.join(af_dir, filename)
    gt_path = os.path.join(ground_truth_dir, filename)
    
    # Check if ground truth file exists
    # Note: Filenames are identical in both directories based on previous context
    if not os.path.exists(gt_path):
        return None
        
    try:
        parser = PDBParser(QUIET=True)
        struct_af = parser.get_structure("af", af_path)
        struct_gt = parser.get_structure("gt", gt_path)
        
        atoms_af_dict = get_ca_atoms(struct_af)
        atoms_gt_dict = get_ca_atoms(struct_gt)
        
        common_keys = set(atoms_af_dict.keys()) & set(atoms_gt_dict.keys())
        
        if len(common_keys) < 3:
            return None
            
        fixed_atoms = []
        moving_atoms = []
        
        sorted_keys = sorted(list(common_keys), key=lambda x: (x[0], x[1][1], x[1][2]))
        
        for k in sorted_keys:
            fixed_atoms.append(atoms_gt_dict[k])
            moving_atoms.append(atoms_af_dict[k])
            
        sup = Superimposer()
        sup.set_atoms(fixed_atoms, moving_atoms)
        
        return {
            "SystemID": system_id,
            "RMSD": sup.rms,
            "CommonResidues": len(common_keys)
        }
        
    except Exception as e:
        print(f"Error processing {system_id}: {e}")
        return None

def main():
    print(f"Calculating RMSD for files in {af_dir} against {ground_truth_dir}...")
    
    if not os.path.exists(af_dir):
        print(f"Error: {af_dir} does not exist")
        exit(1)

    files = sorted([f for f in os.listdir(af_dir) if f.endswith("_protein.pdb")])
    print(f"Found {len(files)} protein files.")

    # Use multiprocessing to speed up processing of ~1000 files
    num_processes = min(cpu_count(), 16) 
    print(f"Using {num_processes} processes...")
    
    with Pool(num_processes) as p:
        results = p.map(process_system, files)
    
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    
    if valid_results:
        df = pd.DataFrame(valid_results)
        df.to_csv(output_csv, index=False)
        print(f"\nDone. Results saved to {output_csv}")
        print(f"Processed {len(valid_results)} systems.")
        print(f"Average RMSD: {df['RMSD'].mean():.4f}")
        print(f"Median RMSD: {df['RMSD'].median():.4f}")
        print(f"Min RMSD: {df['RMSD'].min():.4f}")
        print(f"Max RMSD: {df['RMSD'].max():.4f}")
    else:
        print("No valid results calculated.")

if __name__ == "__main__":
    main()
