#%%

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.join(__file__, '../../..'))
from src.db_utils import db_connection
import pickle
import matplotlib.pyplot as plt
from src.lightning import YuelPocket
import random
from src.datasets import collate
from yuel_pocket import PdbSdfDataset, predict_pocket
from cluster_allosteric import calculate_allosteric_clusters, plot_allosteric_clusters

def get_top_probes():
    """Get all probes from probe set."""
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT lname 
            FROM probe_set 
            ORDER BY id
        """)
        return [row[0] for row in cur.fetchall()]

def analyze_predictions():
    """Analyze pocket predictions for allosteric and active sites."""
    # Get probe ligands
    probe_ligands = get_top_probes(3)
    
    # Store results
    allosteric_scores = []
    active_scores = []
    
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Get all predictions
        cur.execute("""
            SELECT ap.pdb_id, ap.ligand_name, ap.pocket_pred,
                   a.in_allosteric_site, a.in_active_site
            FROM allobench_predictions ap
            JOIN allobench a ON ap.pdb_id = a.pdb_id
            WHERE a.in_allosteric_site IS NOT NULL 
            AND a.in_active_site IS NOT NULL
        """)
        
        for row in cur.fetchall():
            pdb_id, ligand_name, pocket_pred, in_allosteric, in_active = row
            
            try:
                # Convert predictions and site indicators to numpy arrays
                pocket_pred = pickle.loads(pocket_pred)
                in_allosteric = np.array(in_allosteric)
                in_active = np.array(in_active)
                
                # Skip if arrays don't match
                if len(pocket_pred) != len(in_allosteric) or len(pocket_pred) != len(in_active):
                    continue
                
                # Calculate overlap scores
                if len(in_allosteric) > 0 and np.any(in_allosteric == 1):
                    allosteric_score = np.mean(pocket_pred[in_allosteric == 1])
                    allosteric_scores.append(allosteric_score)
                
                if len(in_active) > 0 and np.any(in_active == 1):
                    active_score = np.mean(pocket_pred[in_active == 1])
                    active_scores.append(active_score)
                    
            except Exception as e:
                continue
    
    # Check if we have any valid scores
    if not allosteric_scores or not active_scores:
        print("No valid predictions found!")
        return
    
    # Plot results
    os.makedirs('plots', exist_ok=True)
    
    plt.figure(figsize=(3.5, 2.5))
    plt.boxplot([allosteric_scores, active_scores], labels=['Allosteric Sites', 'Active Sites'])
    plt.ylabel('Mean Prediction Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    print("\nSaving plot to plots/site_predictions.svg")
    plt.savefig('plots/site_predictions.svg')
    plt.show()
    
    # Print statistics
    print("\nStatistics:")
    print(f"Number of valid predictions: {len(allosteric_scores)}")
    print(f"Mean allosteric site score: {np.mean(allosteric_scores):.3f} ± {np.std(allosteric_scores):.3f}")
    print(f"Mean active site score: {np.mean(active_scores):.3f} ± {np.std(active_scores):.3f}")

def analyze_site_counts():
    """Analyze and visualize the number of predicted allosteric and active sites."""
    # Store results
    protein_counts = []  # Will store (protein_id, n_allosteric, n_active)
    
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Get all predictions
        cur.execute("""
            SELECT DISTINCT a.protein_asd_id, 
                   a.in_allosteric_site, a.in_active_site
            FROM allobench a
            WHERE a.in_allosteric_site IS NOT NULL 
            AND a.in_active_site IS NOT NULL
        """)
        
        for row in cur.fetchall():
            protein_id, in_allosteric, in_active = row
            
            try:
                # Convert site indicators to numpy arrays
                in_allosteric = np.array(in_allosteric)
                in_active = np.array(in_active)
                
                # Count sites
                n_allosteric = np.sum(in_allosteric == 1)
                n_active = np.sum(in_active == 1)
                
                protein_counts.append((protein_id, n_allosteric, n_active))
                    
            except Exception as e:
                continue
    
    if not protein_counts:
        print("No valid data found!")
        return
        
    # Convert to numpy arrays for easier analysis
    protein_counts = np.array([(n_allo, n_act) for _, n_allo, n_act in protein_counts])
    n_allosteric = protein_counts[:, 0]
    n_active = protein_counts[:, 1]
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot 1: Distribution of site counts
    plt.figure(figsize=(3.5, 2.5))
    plt.hist([n_allosteric, n_active], label=['Allosteric Sites', 'Active Sites'], 
             bins=20, alpha=0.7, color=['#43a3ef', '#ef767b'])
    plt.xlabel('Number of Sites per Protein')
    plt.ylabel('Frequency')
    plt.legend(frameon=False)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('plots/site_count_distribution.svg')
    print("\nSaving site count distribution to plots/site_count_distribution.svg")
    plt.show()
    
    # Plot 2: Scatter plot of allosteric vs active sites
    plt.figure(figsize=(3.5, 2.5))
    plt.scatter(n_allosteric, n_active, alpha=0.5, color='#43a3ef')
    plt.xlabel('Number of Allosteric Sites')
    plt.ylabel('Number of Active Sites')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('plots/site_count_correlation.svg')
    print("Saving site count correlation to plots/site_count_correlation.svg")
    plt.show()
    
    # Print statistics
    print("\nStatistics:")
    print(f"Total number of proteins analyzed: {len(protein_counts)}")
    print(f"Average number of allosteric sites per protein: {np.mean(n_allosteric):.2f} ± {np.std(n_allosteric):.2f}")
    print(f"Average number of active sites per protein: {np.mean(n_active):.2f} ± {np.std(n_active):.2f}")
    print(f"Correlation between allosteric and active site counts: {np.corrcoef(n_allosteric, n_active)[0,1]:.3f}")
    print(f"\nRange of allosteric sites: {np.min(n_allosteric):.0f} - {np.max(n_allosteric):.0f}")
    print(f"Range of active sites: {np.min(n_active):.0f} - {np.max(n_active):.0f}")

def analyze_pocket_coverage():
    """Analyze how the number of residues predicted as pockets changes with increasing number of probes."""
    # Store coverage counts for each protein at each n_probes
    coverage_by_n_probes = {}  # n_probes -> list of coverage counts
    
    # Get all predictions
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT ap.pdb_id, ap.ligand_name, ap.pocket_pred
            FROM allobench_predictions ap
            JOIN allobench a ON ap.pdb_id = a.pdb_id
            WHERE a.in_allosteric_site IS NOT NULL 
            AND a.modulator_class = 'Lig'
            AND array_length(in_allosteric_site, 1) < 2000
        """)
        
        # Group predictions by protein
        protein_results = {}
        for row in cur.fetchall():
            pdb_id, ligand_name, pocket_pred = row
            
            if pdb_id not in protein_results:
                protein_results[pdb_id] = {'probes': {}}
            
            try:
                pred = pickle.loads(pocket_pred)
                protein_results[pdb_id]['probes'][ligand_name] = pred
            except Exception as e:
                print(f"Error processing {pdb_id}: {str(e)}")
                continue
    
    # Get maximum number of probes
    max_probes = max(len(p['probes']) for p in protein_results.values())
    
    # Initialize storage for each number of probes
    for n in range(1, max_probes + 1):
        coverage_by_n_probes[n] = []
    
    # Calculate coverage counts
    for protein_data in protein_results.values():
        probe_results_dict = protein_data['probes']
        if len(probe_results_dict) < max_probes:
            continue
            
        # Get protein length from first probe result
        first_probe = list(probe_results_dict.values())[0]
        n_residues = len(first_probe)
        
        # Calculate coverage for different numbers of probes
        for n_probes in range(1, max_probes + 1):
            # Initialize combined predictions mask
            combined_pred_mask = np.zeros(n_residues, dtype=bool)
            
            # Use first n_probes probes
            probe_names = list(probe_results_dict.keys())[:n_probes]
            for probe_name in probe_names:
                pred = probe_results_dict[probe_name]
                
                # Calculate IQR threshold for this probe
                q1 = np.percentile(pred, 25)
                q3 = np.percentile(pred, 75)
                iqr = q3 - q1
                iqr_threshold = q3 + 1.5 * iqr
                
                # Update combined mask (OR operation)
                combined_pred_mask = combined_pred_mask | (pred > iqr_threshold)
            
            # Calculate and store coverage count
            coverage_count = np.sum(combined_pred_mask)
            coverage_by_n_probes[n_probes].append(coverage_count)
    
    # Calculate statistics
    mean_coverage = []
    std_coverage = []
    for n in range(1, max_probes + 1):
        counts = np.array(coverage_by_n_probes[n])
        mean_coverage.append(np.mean(counts))
        std_coverage.append(np.std(counts))
    
    # Plot results
    plt.figure(figsize=(3.5, 2.5))
    x = range(1, max_probes + 1)
    plt.plot(x, mean_coverage, color='#ef767b', linewidth=2)
    plt.fill_between(x, 
                     np.array(mean_coverage) - np.array(std_coverage),
                     np.array(mean_coverage) + np.array(std_coverage),
                     color='#ef767b', alpha=0.2)
    
    plt.xlabel('Number of Probes')
    plt.ylabel('Number of Predicted Pocket Residues')
    plt.grid(True, linestyle='--', alpha=0.7)
    print("\nSaving plot to plots/pocket_coverage.svg")
    plt.savefig('plots/pocket_coverage.svg')
    plt.show()
    
    # Print statistics
    print("\nPocket Coverage Statistics:")
    print(f"Initial coverage (1 probe): {mean_coverage[0]:.1f} ± {std_coverage[0]:.1f} residues")
    print(f"Final coverage ({max_probes} probes): {mean_coverage[-1]:.1f} ± {std_coverage[-1]:.1f} residues")
    print(f"Absolute increase: {mean_coverage[-1] - mean_coverage[0]:.1f} residues")

def analyze_prediction_ratios(nprobes):
    """Analyze pocket predictions for allosteric sites by combining predictions from all probes."""
    # Store results by protein
    protein_results = {}  # Will store {allobench_id: {probe_name: (pocket_pred, in_allosteric)}}
    
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Get all predictions for proteins with Lig modulators
        cur.execute("""
            SELECT a.id, a.pdb_id, ap.ligand_name, ap.pocket_pred,
                   a.in_allosteric_site
            FROM allobench_predictions ap
            JOIN allobench a ON ap.pdb_id = a.pdb_id
            WHERE a.in_allosteric_site IS NOT NULL 
            AND a.modulator_class = 'Lig'
            AND array_length(in_allosteric_site, 1) < 2000
        """)
        
        for row in cur.fetchall():
            allobench_id, pdb_id, ligand_name, pocket_pred, in_allosteric = row
            
            try:
                # Convert predictions and site indicators to numpy arrays
                pocket_pred = pickle.loads(pocket_pred)
                in_allosteric = np.array(in_allosteric)
                
                # Only use protein residue predictions (exclude ligand atoms)
                n_residues = len(in_allosteric)  # This is the number of protein residues
                if n_residues == 0:
                    continue
                pocket_pred = pocket_pred[:n_residues]  # Only take predictions for protein residues

                # Store results by protein
                if allobench_id not in protein_results:
                    protein_results[allobench_id] = {
                        'pdb_id': pdb_id,
                        'probes': {}
                    }
                protein_results[allobench_id]['probes'][ligand_name] = (pocket_pred, in_allosteric)
                    
            except Exception as e:
                print(f"Error processing {pdb_id}: {str(e)}")
                continue
    
    if not protein_results:
        print("No valid predictions found!")
        return {}
        
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Store results for different numbers of probes
    results_by_n_probes = {i: [] for i in range(1, nprobes+1)}  # n_probes -> list of ratios
    final_results = {}  # Store protein_asd_id -> (ratio, combined_pred_mask, pdb_id)
    
    # Get list of probe names
    probe_names = list(next(iter(protein_results.values()))['probes'].keys())
    
    for allobench_id, data in protein_results.items():
        probe_results_dict = data['probes']
        # Skip proteins that don't have all probes
        if len(probe_results_dict) < nprobes:  # Need at least 5 probes
            continue
            
        # Get predictions and site indicators (should be same for all probes)
        first_probe = list(probe_results_dict.values())[0]
        in_allosteric = first_probe[1]
        n_residues = len(in_allosteric)
        
        # Test with different numbers of probes
        for n_probes in range(1, nprobes+1):
            # Initialize combined predictions mask
            combined_pred_mask = np.zeros(n_residues, dtype=bool)
            
            # Use first n_probes probes
            for probe_name in probe_names[:n_probes]:
                pred, _ = probe_results_dict[probe_name]

                if len(pred) < n_residues:
                    print(allobench_id, data['pdb_id'], probe_name, len(pred), n_residues)
                    raise Exception('Prediction length mismatch')
                
                # Check for NaN in predictions
                if np.any(np.isnan(pred)):
                    continue
                    
                # Calculate IQR threshold for this probe
                q1 = np.percentile(pred, 25)
                q3 = np.percentile(pred, 75)
                iqr = q3 - q1
                iqr_threshold = q3 + 1.5 * iqr
                
                # Update combined mask (OR operation)
                combined_pred_mask = combined_pred_mask | (pred > iqr_threshold)
            
            # Calculate and store ratio for this number of probes
            if np.any(in_allosteric == 1):
                ratio = np.mean(combined_pred_mask[in_allosteric == 1])
                results_by_n_probes[n_probes].append(ratio)
                
                # Store final results (using all probes) for structure saving
                if n_probes == nprobes:
                    final_results[allobench_id] = {
                        'ratio': ratio,
                        'mask': combined_pred_mask,
                        'pdb_id': data['pdb_id'],
                        'in_allosteric': in_allosteric
                    }
    
    # Calculate success rate curves
    thresholds = np.linspace(0, 1, 100)
    
    # Plot success rate vs ratio threshold
    plt.figure(figsize=(3.5, 2.5))
    
    # Generate colors and linestyles dynamically
    base_color = np.array([239/255, 118/255, 123/255])  # #ef767b in RGB
    colors = []
    for i in range(nprobes-1):
        # Create gradient by adjusting the saturation
        factor = 0.4 + 0.6 * (i / (nprobes-1))
        color = base_color * factor + (1 - factor)
        colors.append(color.tolist() + [1.0])  # Add alpha channel
    colors = np.array(colors)
    colors = np.vstack((colors, [0.2, 0.2, 0.2, 1]))  # Add black for the last probe
    linestyles = ['-'] * nprobes  # All solid lines
    
    for n_probes, color, linestyle in zip(range(1, nprobes+1), colors, linestyles):
        ratios = np.array(results_by_n_probes[n_probes])
        ratios = ratios[~np.isnan(ratios)]
        success_rates = [np.mean(ratios >= t) * 100 for t in thresholds]
        plt.plot(thresholds, success_rates, color=color if isinstance(color, str) else color.tolist(),
                linewidth=2 if n_probes == nprobes else 1.5,
                linestyle=linestyle,
                label=f'{n_probes} probe{"s" if n_probes > 1 else ""}')
    
    plt.xlabel('Ratio Threshold')
    plt.ylabel('Success Rate (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(frameon=False)
    print("\nSaving plot to plots/allosteric_site_success_rate.svg")
    plt.savefig('plots/allosteric_site_success_rate.svg')
    plt.show()
    
    # Print summary statistics
    print("\nSummary statistics for allosteric site predictions:")
    for n_probes in range(1, nprobes+1):
        ratios = np.array(results_by_n_probes[n_probes])
        ratios = ratios[~np.isnan(ratios)]
        print(f"\n{n_probes} probe{'s' if n_probes > 1 else ''}:")
        print(f"Number of proteins: {len(ratios)}")
        print(f"Mean ratio: {np.mean(ratios):.3f} ± {np.std(ratios):.3f}")
        print(f"Success rate at 0.5 threshold: {np.mean(ratios >= 0.5)*100:.1f}%")
    
    return final_results

def save_structures_with_bfactors(results):
    """Save structures with pocket predictions and both allosteric and active sites as bfactors."""
    from src.clustering import write_bfactor_to_pdb
    from src.pdb_utils import Structure, Model, Chain
    from io import StringIO
    
    if not results:
        print("No results to process!")
        return
        
    # Find proteins with low ratios (bottom 25%)
    ratios = np.array([r['ratio'] for r in results.values()])
    low_ratio_threshold = np.percentile(ratios, 25)
    low_ratio_proteins = [(pid, data) for pid, data in results.items() 
                         if data['ratio'] <= low_ratio_threshold]
    
    # Randomly select one protein
    allobench_id, data = random.choice(low_ratio_proteins)
    pdb_id = data['pdb_id']
    
    # Get PDB content and active site information
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT pdb_content, in_active_site
            FROM allobench
            WHERE id = %s
        """, (allobench_id,))
        pdb_content, in_active_site = cur.fetchone()
    
    # Parse PDB and create filtered structure
    structure = Structure()
    structure.read(StringIO(pdb_content))
    
    # Create new structure with only CA-containing residues
    new_structure = Structure()
    new_model = Model()
    new_structure.models.append(new_model)
    
    for residue in structure[0].get_residues():
        if any(atom.atom_name == "CA" for atom in residue.atoms):
            if residue.chain_id not in [c.chain_id for c in new_model.chains]:
                new_model.add_chain(Chain(residue.chain_id))
            chain = next(c for c in new_model.chains if c.chain_id == residue.chain_id)
            chain.add_residue(residue)
    
    # Save structures
    os.makedirs('structures', exist_ok=True)
    
    # Save pocket predictions
    write_bfactor_to_pdb(data['mask'].astype(float), new_structure.to_pdb(), 
                        f'structures/{pdb_id}_pocket_pred.pdb')
    
    # Save allosteric sites
    write_bfactor_to_pdb(data['in_allosteric'].astype(float), new_structure.to_pdb(), 
                        f'structures/{pdb_id}_allosteric_sites.pdb')
    
    # Save active sites
    write_bfactor_to_pdb(np.array(in_active_site).astype(float), new_structure.to_pdb(), 
                        f'structures/{pdb_id}_active_sites.pdb')
    
    print(f"\nSaved structures for {pdb_id} (allosteric ratio: {data['ratio']:.3f}) in structures/ directory:")
    print(f"- Pocket predictions: structures/{pdb_id}_pocket_pred.pdb")
    print(f"- Allosteric sites: structures/{pdb_id}_allosteric_sites.pdb")
    print(f"- Active sites: structures/{pdb_id}_active_sites.pdb")

def analyze_protein_sizes():
    """Analyze and visualize the distribution of protein sizes in the allobench dataset."""
    # Store protein sizes
    protein_sizes = {}  # pdb_id -> size
    
    with db_connection() as conn:
        cur = conn.cursor()
        
        # Get all unique proteins
        cur.execute("""
            SELECT DISTINCT ON (pdb_id) pdb_id, in_active_site
            FROM allobench
            WHERE in_active_site IS NOT NULL
        """)
        
        for row in cur.fetchall():
            pdb_id, in_active_site = row
            
            try:
                # Convert to numpy array and get size
                size = len(np.array(in_active_site))
                protein_sizes[pdb_id] = size
                    
            except Exception as e:
                print(f"Error processing {pdb_id}: {str(e)}")
                continue
    
    if not protein_sizes:
        print("No valid data found!")
        return
        
    # Convert to numpy array for analysis
    sizes = np.array(list(protein_sizes.values()))
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot size distribution
    plt.figure(figsize=(3.5, 2.5))
    plt.hist(sizes, bins=30, color='#43a3ef', alpha=0.7)
    plt.xlabel('Protein Size (Number of Residues)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    print("\nSaving protein size distribution to plots/protein_size_distribution.svg")
    plt.savefig('plots/protein_size_distribution.svg')
    plt.show()
    
    # Print statistics
    print("\nProtein Size Statistics:")
    print(f"Total number of unique proteins: {len(sizes)}")
    print(f"Mean protein size: {np.mean(sizes):.1f} ± {np.std(sizes):.1f} residues")
    print(f"Median protein size: {np.median(sizes):.1f} residues")
    print(f"Size range: {np.min(sizes):.0f} - {np.max(sizes):.0f} residues")

def analyze_allosteric_ratio_distribution():
    """Analyze and plot the distribution of number of allosteric site residues in proteins."""
    # Get allosteric site counts
    counts = []
    
    with db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT ON (pdb_id) pdb_id, in_allosteric_site
            FROM allobench
            WHERE in_allosteric_site IS NOT NULL 
            AND modulator_class = 'Lig'
            AND array_length(in_allosteric_site, 1) < 2000
        """)
        
        for row in cur.fetchall():
            pdb_id, in_allosteric = row
            try:
                # Convert to numpy array and calculate count
                in_allosteric = np.array(in_allosteric)
                count = np.sum(in_allosteric == 1)
                counts.append(count)
            except Exception as e:
                print(f"Error processing {pdb_id}: {str(e)}")
                continue
    
    if not counts:
        print("No valid data found!")
        return
    
    counts = np.array(counts)
    
    # Create density plot
    plt.figure(figsize=(3.5, 2.5))
    
    # Calculate histogram
    hist_counts, bins = np.histogram(counts, bins=30, density=True)
    # Calculate bin centers for plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Plot smoothed line
    plt.plot(bin_centers, hist_counts, color='#ef767b', linewidth=2)
    
    plt.xlabel('Number of Allosteric Site Residues')
    plt.ylabel('Density')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Print statistics
    print("\nAllosteric Site Statistics:")
    print(f"Number of proteins: {len(counts)}")
    print(f"Mean number of residues: {np.mean(counts):.1f} ± {np.std(counts):.1f}")
    print(f"Median number of residues: {np.median(counts):.1f}")
    print(f"Range: {np.min(counts):.0f} - {np.max(counts):.0f} residues")
    
    print("\nSaving plot to plots/allosteric_site_distribution.svg")
    plt.savefig('plots/allosteric_site_distribution.svg')
    plt.show()

if __name__ == "__main__":
    
    # Then analyze predictions
    # analyze_predictions()
    
    # Analyze site counts
    # analyze_site_counts()
    
    # Analyze prediction ratios
    # results = analyze_prediction_ratios(3)
    
    # Analyze pocket coverage
    # analyze_pocket_coverage()
    
    # Analyze allosteric site ratio distribution
    # analyze_allosteric_ratio_distribution()
    
    # Save structures with bfactors
    # save_structures_with_bfactors(results)
    
    # Analyze protein sizes
    # analyze_protein_sizes()

# %%
