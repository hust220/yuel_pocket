import os
import json
import warnings
import sys
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Superimposer, Structure, Model, Chain
from Bio import Align
from Bio.SeqUtils import seq1
import numpy as np
from tqdm import tqdm
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# Suppress PDB warnings
warnings.simplefilter('ignore', PDBConstructionWarning)

def get_filtered_residues(chain):
    """Get list of standard residues from a chain."""
    return [res for res in chain if res.get_id()[0] == " "]

def get_sequence(residues):
    """Extract 1-letter sequence from a list of residues."""
    seq = ""
    for res in residues:
        try:
            char = seq1(res.get_resname())
            if not char or char == " ": 
                char = "X"
            seq += char
        except:
            seq += "X"
    return seq

def align_af_models_to_test1036():
    # Root paths
    base_dir = Path("/home/tyq4zn/scratch/codes/yuel_pocket/benchmark/alphafold")
    test_dir = base_dir / "test1036"
    af_dir = base_dir / "af_models"
    output_dir = base_dir / "test1036_af_aligned"
    output_dir.mkdir(exist_ok=True)
    
    mapping_file = base_dir / "pdb_uniprot_af_mapping.json"
    if not mapping_file.exists():
        print(f"Error: mapping file not found at {mapping_file}")
        return
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    parser = PDBParser(QUIET=True)
    
    # Initialize Aligner
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -0.5
    aligner.extend_gap_score = -0.1

    protein_files = sorted(list(test_dir.glob("*_protein.pdb")))
    if not protein_files:
        print(f"No protein files found in {test_dir}")
        return
        
    print(f"Aligning AF models for {len(protein_files)} systems...")
    
    for prot_path in tqdm(protein_files):
        system_id = prot_path.name.replace("_protein.pdb", "")
        # Get PDB ID (starts with 4 chars)
        pdb_id = prot_path.name.split('__')[0].lower()
        
        try:
            exp_struct = parser.get_structure('exp', str(prot_path))
        except Exception as e:
            # print(f"Error parsing {prot_path}: {e}")
            continue
            
        uniprot_ids = mapping.get(pdb_id, [])
        if not uniprot_ids:
            continue
            
        # Load candidate AF structures for this PDB ID
        af_structs = {}
        for upid in uniprot_ids:
            af_path = af_dir / f"{pdb_id}_{upid}_AF.pdb"
            if af_path.exists():
                try:
                    af_structs[upid] = parser.get_structure(upid, str(af_path))
                except:
                    continue
        
        if not af_structs:
            continue

        # Create new structure for the aligned AF model
        new_struct = Structure.Structure(system_id)
        new_model = Model.Model(0)
        new_struct.add(new_model)
        
        chains_mapped = 0
        
        # Match each chain in Experimental Protein
        for exp_chain in exp_struct.get_chains():
            exp_residues = get_filtered_residues(exp_chain)
            exp_seq = get_sequence(exp_residues)
            if not exp_seq:
                continue
            
            best_upid = None
            best_score = -1e9
            best_alignment = None
            
            for upid, af_struct in af_structs.items():
                af_chain = list(af_struct.get_chains())[0]
                af_residues = get_filtered_residues(af_chain)
                af_seq = get_sequence(af_residues)
                if not af_seq: continue
                
                try:
                    score = aligner.score(af_seq, exp_seq)
                    if score > best_score:
                        # Only get actual alignment if it's the best so far
                        alignments = aligner.align(af_seq, exp_seq)
                        # Avoid if alignments: or len(alignments)
                        try:
                            top_align = alignments[0]
                            best_score = score
                            best_upid = upid
                            best_alignment = top_align
                        except (OverflowError, IndexError):
                            # Fallback or skip if alignment object is too complex
                            continue
                except Exception:
                    continue
            
            # Identity check (heuristic)
            # score is typically 2*matches if perfect
            if best_score < 1.0 * len(exp_seq) or best_alignment is None:
                continue
                
            # Perform superposition
            ref_af_struct = af_structs[best_upid]
            af_chain_res = get_filtered_residues(list(ref_af_struct.get_chains())[0])
            
            # Map AF residues to Experimental residues based on alignment
            # best_alignment.aligned returns segments: ( ( (s1, e1), ... ), ( (s2, e2), ... ) )
            seg_af, seg_exp = best_alignment.aligned
            
            mapping_pairs = []
            for s1, s2 in zip(seg_af, seg_exp):
                for i in range(s1[1] - s1[0]):
                    mapping_pairs.append((s1[0] + i, s2[0] + i))
            
            moving_atoms = [] # AF
            fixed_atoms = []  # Exp
            
            for af_idx, exp_idx in mapping_pairs:
                if af_idx < len(af_chain_res) and exp_idx < len(exp_residues):
                    res_af = af_chain_res[af_idx]
                    res_exp = exp_residues[exp_idx]
                    if 'CA' in res_af and 'CA' in res_exp:
                        moving_atoms.append(res_af['CA'])
                        fixed_atoms.append(res_exp['CA'])
            
            if len(fixed_atoms) < 3:
                continue
                
            # Copy AF structure to avoid modifying shared references
            # Use fresh parse to ensure no prior transformations affect it
            af_path = af_dir / f"{pdb_id}_{best_upid}_AF.pdb"
            final_af_struct = parser.get_structure('final', str(af_path))
            
            superimposer = Superimposer()
            superimposer.set_atoms(fixed_atoms, moving_atoms)
            superimposer.apply(final_af_struct.get_atoms())
            
            # Construct the aligned chain using AF coordinates but Exp sequence alignment
            aligned_chain = Chain.Chain(exp_chain.id)
            final_af_res = get_filtered_residues(list(final_af_struct.get_chains())[0])
            
            for af_idx, exp_idx in mapping_pairs:
                res_af = final_af_res[af_idx]
                res_exp = exp_residues[exp_idx]
                # Clone AF residue, set ID to match Exp residue
                res_copy = res_af.copy()
                res_copy.id = res_exp.id
                aligned_chain.add(res_copy)
            
            if len(aligned_chain) > 0:
                new_model.add(aligned_chain)
                chains_mapped += 1
        
        if chains_mapped > 0:
            out_path = output_dir / f"{system_id}_af_aligned.pdb"
            io = PDBIO()
            io.set_structure(new_struct)
            io.save(str(out_path))

    print(f"\nAlignment finished. Aligned AF models are in {output_dir}")

if __name__ == "__main__":
    align_af_models_to_test1036()
