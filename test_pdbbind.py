#%%
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDetermineBonds
from tqdm import tqdm
from anal import plot_metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from anal import plot_metrics
from rdkit.RDLogger import DisableLog
DisableLog('rdApp.*')

#%%
def calculate_metrics(mol1, mol2):
    # according to the coordinates to create the mapping of the index of atoms in mol1 to mol2
    # get coordinates from conformer 0
    c1 = mol1.GetConformer(0).GetPositions()
    c2 = mol2.GetConformer(0).GetPositions()
    mapping = {}
    order = []
    for i, atom1 in enumerate(mol1.GetAtoms()):
        for j, atom2 in enumerate(mol2.GetAtoms()):
            if np.linalg.norm(c1[i] - c2[j]) < 1e-2:
                mapping[i] = j
                order.append(j)
                break

    # reorder the bonds of mol2 according to the mapping
    bonds1 = [(bond.GetBeginAtomIdx(),bond.GetEndAtomIdx(),bond.GetBondType()) for bond in mol1.GetBonds()]
    bonds2 = [(bond.GetBeginAtomIdx(),bond.GetEndAtomIdx(),bond.GetBondType()) for bond in mol2.GetBonds()]
    # print(mapping)
    # print(bonds1)
    # print(bonds2)
    bonds1 = [(mapping[bond[0]],mapping[bond[1]],bond[2]) for bond in bonds1]

    # make sure i<j for all bonds
    bonds1 = [(min(bond[0],bond[1]),max(bond[0],bond[1]),bond[2]) for bond in bonds1]
    bonds2 = [(min(bond[0],bond[1]),max(bond[0],bond[1]),bond[2]) for bond in bonds2]

    # reorder bonds1 and bonds2 with asceding order of the first element and second element
    bonds1 = sorted(bonds1, key=lambda x: (x[0], x[1]))
    bonds2 = sorted(bonds2, key=lambda x: (x[0], x[1]))

    # Get the bond order prediction from the original molecule
    bond_order_true = np.array([bond[2] for bond in bonds1])
    bond_order_pred = np.array([bond[2] for bond in bonds2])  

#    print(bond_order_true)
#    print(bond_order_pred)

    # check if number of bonds are the same
    if len(bond_order_pred) != len(bond_order_true):
        raise ValueError('Number of bonds are not the same')

    accuracy = accuracy_score(bond_order_true, bond_order_pred)
    precision = precision_score(bond_order_true, bond_order_pred, average='weighted', zero_division=0)
    recall = recall_score(bond_order_true, bond_order_pred, average='weighted', zero_division=0)
    f1 = f1_score(bond_order_true, bond_order_pred, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1

# read the two sdf files
supplier1 = Chem.SDMolSupplier('data/pdbbind/pdbbind_ligands_origin.sdf')
supplier2 = Chem.SDMolSupplier('data/pdbbind/pdbbind_ligands.sdf')

# calculate the metrics
accuracies = []
precisions = []
recalls = []
f1_scores = []
nfails = 0
npasses = 0
for mol1, mol2 in zip(supplier1, supplier2):
    # calculate the metrics
    try:
        if mol1 is None or mol2 is None:
            continue
        accuracy, precision, recall, f1 = calculate_metrics(mol1, mol2)
#        print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 score: {f1_score}')
        # if every bond is correct, then it is a pass
        if f1 == 1:
            npasses += 1
    except ValueError as e:
        # tqdm.write(f'{e}')
        nfails += 1
        continue
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
plot_metrics(accuracies, precisions, recalls, f1_scores, outfile='figures/pdbbind_metrics.svg')
print(f'Accuracy: {sum(accuracies) / len(accuracies)}')
print(f'Precision: {sum(precisions) / len(precisions)}')
print(f'Recall: {sum(recalls) / len(recalls)}')
print(f'F1 score: {sum(f1_scores) / len(f1_scores)}')
print(f'Statistics: Fails: {nfails}/{len(supplier1)}={nfails / len(supplier1)}, Perfects: {npasses}/{len(supplier1)}={npasses / len(supplier1)}')

# %%
