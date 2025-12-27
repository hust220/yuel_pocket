import os
import requests
import json
from pathlib import Path
from tqdm import tqdm

def get_uniprot_ids(pdb_id):
    """Get UniProt IDs associated with a PDB ID using PDBe API."""
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if pdb_id in data:
                uniprot_ids = list(data[pdb_id]['UniProt'].keys())
                return uniprot_ids
    except Exception as e:
        print(f"Error fetching mapping for {pdb_id}: {e}")
    return []

def download_af_model(uniprot_id, output_path):
    """Download AlphaFold model for a UniProt ID using AF API."""
    api_url = f"https://www.alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list):
                # Use the first entry's pdbUrl
                pdb_url = data[0].get('pdbUrl')
                if pdb_url:
                    pdb_response = requests.get(pdb_url, timeout=10)
                    if pdb_response.status_code == 200:
                        with open(output_path, 'wb') as f:
                            f.write(pdb_response.content)
                        return True
    except Exception as e:
        print(f"Error downloading AF model for {uniprot_id}: {e}")
    return False

def main(limit=None):
    test_dir = Path("test1036")
    output_dir = Path("af_models")
    output_dir.mkdir(exist_ok=True)
    
    mapping_file = Path("pdb_uniprot_af_mapping.json")
    mapping = {}
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)

    # 1. Identify PDB IDs
    protein_files = list(test_dir.glob("*_protein.pdb"))
    pdb_ids = set()
    for f in protein_files:
        # system_id is before _protein.pdb
        # pdb_id is before the first __
        pdb_id = f.name.split('__')[0].lower()
        pdb_ids.add(pdb_id)
    
    sorted_pdb_ids = sorted(list(pdb_ids))
    if limit:
        sorted_pdb_ids = sorted_pdb_ids[:limit]
        print(f"Testing with limit of {limit} PDB IDs")
    else:
        print(f"Found {len(pdb_ids)} unique PDB IDs in {test_dir}")

    # 2. Process each PDB ID
    for pdb_id in tqdm(sorted_pdb_ids, desc="Fetching AF models"):
        if pdb_id in mapping and mapping[pdb_id]:
            uniprot_ids = mapping[pdb_id]
        else:
            uniprot_ids = get_uniprot_ids(pdb_id)
            mapping[pdb_id] = uniprot_ids
        
        for upid in uniprot_ids:
            # Note: we don't know the version beforehand now, so just save as .pdb
            af_filename = f"{pdb_id}_{upid}_AF.pdb"
            af_path = output_dir / af_filename
            if not af_path.exists():
                download_af_model(upid, af_path)
        
        # Save mapping periodically
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=4)

    print(f"\nFinished. AF models are in {output_dir}")
    print(f"Mapping saved to {mapping_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(limit=args.limit)
