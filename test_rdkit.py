#%%

# read test_results/geom_kekulized_test_true_bonds.sdf
# use rdkit to rebuild the bonds of each molecue and save the sdf file

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds
from tqdm import tqdm
from anal import plot_metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from yuel_bond import append_mol_to_sdf

def rebuild_bonds(m):
    # deep copy the molecule
    # mol = Chem.RWMol(m)  # Add hydrogens
    mol = Chem.RWMol(Chem.AddHs(m, addCoords=True))  # Add hydrogens
    # AllChem.EmbedMolecule(mol)  # Generate 2D coordinates
    # AllChem.EmbedMolecule(mol, maxAttempts=10)

    # Now remove all bonds (for demonstration)
    for bond in list(mol.GetBonds()):
        mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    # Determine bonds from coordinates
    rdDetermineBonds.DetermineBonds(mol)

    mol =Chem.RemoveHs(mol)

    # The molecule now has bonds assigned based on atomic distances
    return mol

def test_rdkit_rebuild_bonds(infile, outfile1, outfile2):
    suppl = Chem.SDMolSupplier(infile, sanitize=False, strictParsing=False)
    # clear the outfiles
    open(outfile1, 'w').close()
    open(outfile2, 'w').close()
    for mol in tqdm(suppl, total=len(suppl), desc='Rebuilding bonds'):
        try:
            mol2 = rebuild_bonds(mol)
            if mol is not None and mol2 is not None:
                append_mol_to_sdf(mol2, outfile2)
                append_mol_to_sdf(mol, outfile1)
        except Exception as e:
            tqdm.write(f'{e}')
            continue

# test_rdkit_rebuild_bonds(
#     'test_results/geom_kekulized_test_true.sdf',
#     'test_results/geom_kekulized_test_true_rdkit_original.sdf',
#     'test_results/geom_kekulized_test_true_rdkit_rebuilt.sdf')
test_rdkit_rebuild_bonds(
    'test_results/geom_sanitized_test_noise_0_2_true.sdf',
    'test_results/geom_sanitized_test_noise_0_2_rdkit_original.sdf',
    'test_results/geom_sanitized_test_noise_0_2_rdkit_rebuilt.sdf')

#%%
# read the two sdf files and calculate Accuracy, Precision, Recall, and F1 score of bond order prediction
from rdkit.Chem import AllChem
from rdkit import Chem
import numpy as np
from tqdm import tqdm
from anal import plot_metrics
import matplotlib.pyplot as plt
def calculate_metrics(mol1, mol2):
    # according to the coordinates to create the mapping of the index of atoms in mol1 to mol2
    # get coordinates from conformer 0
    c1 = mol1.GetConformer(0).GetPositions()
    c2 = mol2.GetConformer(0).GetPositions()
    mapping = {}
    order = []
    for i, atom1 in enumerate(mol1.GetAtoms()):
        for j, atom2 in enumerate(mol2.GetAtoms()):
            if np.linalg.norm(c1[i] - c2[j]) < 1e-4:
                mapping[i] = j
                order.append(j)
                break
    
    # reorder the bonds of mol2 according to the mapping
    bonds1 = [(bond.GetBeginAtomIdx(),bond.GetEndAtomIdx(),bond.GetBondType()) for bond in mol1.GetBonds()]
    bonds2 = [(bond.GetBeginAtomIdx(),bond.GetEndAtomIdx(),bond.GetBondType()) for bond in mol2.GetBonds()]
    bonds1 = [(mapping[bond[0]],mapping[bond[1]],bond[2]) for bond in bonds1]

    bonds1 = {tuple(sorted(bond[0:2])): bond[2] for bond in bonds1}
    bonds2 = {tuple(sorted(bond[0:2])): bond[2] for bond in bonds2}

    # if key is in bonds1 but not in bonds2, then add key,-1 to bonds2
    # if key is in bonds2 but not in bonds1, then add key,-1 to bonds1
    for key in bonds1.keys():
        if key not in bonds2:
            bonds2[key] = -1
    for key in bonds2.keys():
        if key not in bonds1:
            bonds1[key] = -1

    # Get the bond order prediction from the original molecule
    bond_order_true = np.array([bonds1[key] for key in sorted(bonds1.keys())])
    bond_order_pred = np.array([bonds2[key] for key in sorted(bonds2.keys())])  

#    print(bond_order_true)
#    print(bond_order_pred)

    accuracy = accuracy_score(bond_order_true, bond_order_pred)
    precision = precision_score(bond_order_true, bond_order_pred, average='weighted', zero_division=0)
    recall = recall_score(bond_order_true, bond_order_pred, average='weighted', zero_division=0)
    f1 = f1_score(bond_order_true, bond_order_pred, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1

def evaluate_sdf_bonds(infile1, infile2, outfile):
    # read the two sdf files
    supplier1 = Chem.SDMolSupplier(infile1, sanitize=False, strictParsing=False)
    supplier2 = Chem.SDMolSupplier(infile2, sanitize=False, strictParsing=False)

    # calculate the metrics
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    nperfect = 0
    nfails = 0
    for mol1, mol2 in zip(supplier1, supplier2):
        # calculate the metrics
        try:
            if mol1 is not None and mol2 is not None:
                accuracy, precision, recall, f1 = calculate_metrics(mol1, mol2)
                # if every bond is correct, then it is a pass
                accuracies.append(accuracy)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
                if f1 == 1:
                    nperfect += 1
                continue
        except ValueError as e:
            # tqdm.write(f'{e}')
            pass
        nfails += 1
        continue
    return {
        'accuracies': accuracies,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
        'nfails': nfails,
        'nperfect': nperfect,
        'nsamples': len(supplier1)
    }

def plot_metrics(metrics1, metrics2, title='', outfile=''):
    nsamples1 = metrics1['nsamples']
    nsamples2 = metrics2['nsamples']

    accuracies1 = metrics1['accuracies']
    precisions1 = metrics1['precisions']
    recalls1 = metrics1['recalls']
    f1_scores1 = metrics1['f1_scores']

    accuracies2 = metrics2['accuracies']
    precisions2 = metrics2['precisions']
    recalls2 = metrics2['recalls']
    f1_scores2 = metrics2['f1_scores']

    values1 = [np.mean(accuracies1), np.mean(precisions1), np.mean(recalls1), np.mean(f1_scores1)]
    stds1 = [np.std(accuracies1), np.std(precisions1), np.std(recalls1), np.std(f1_scores1)]
    
    values2 = [np.mean(accuracies2), np.mean(precisions2), np.mean(recalls2), np.mean(f1_scores2)]
    stds2 = [np.std(accuracies2), np.std(precisions2), np.std(recalls2), np.std(f1_scores2)]
    
    plt.figure(figsize=(2.5,2.5))
    # Capitalize the first letter of each key
    labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    # face colors: white, gray, #2c939a, blue
    # facecolors = ['white', 'gray', '#2c939a', 'black']
    # edge color: #000000, face color: 
    # set x positions of x axis
    x = np.arange(len(labels))
    bars1 = plt.bar(x - 0.2, values1, width=0.4, yerr=stds1, edgecolor='#000000', facecolor='#2c939a', label='YuelBond')
    bars2 = plt.bar(x + 0.2, values2, width=0.4, yerr=stds2, edgecolor='#000000', facecolor='white', label='rdkit')

    # for i, bar in enumerate(bars1):
    #     bar.set_facecolor(facecolors[i])
    # for i, bar in enumerate(bars2):
    #     bar.set_facecolor(facecolors[i])
    if title:
        plt.title(title)
    # legend on the position (0.5, 1.1), 2 columns, no frame
    plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.3), frameon=False)
    # set labels as x tick labels
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()
    plt.clf()
#%%

rdkit_metrics = evaluate_sdf_bonds(
    'test_results/geom_sanitized_test_noise_0_2_rdkit_original.sdf',
    'test_results/geom_sanitized_test_noise_0_2_rdkit_rebuilt.sdf',
    'analyses/figures/rdkit_metrics_noise_0_2.svg')
yuel_metrics = evaluate_sdf_bonds(
    'test_results/geom_sanitized_test_noise_0_2_true.sdf',
    'test_results/geom_sanitized_test_noise_0_2_predictions.sdf',
    'analyses/figures/yuel_metrics_sanitized_noise_0_2.svg')

#%%

# plot_metrics(rdkit_metrics, yuel_metrics, title='Rdkit vs Yuel', outfile='analyses/figures/rdkit_vs_yuel_noise_0_2.svg')
plot_metrics(yuel_metrics, rdkit_metrics)
# %%
