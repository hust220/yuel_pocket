import torch
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import glob
import random

from sklearn.decomposition import PCA
from src import const
from src.molecule_builder import get_bond_order
from rdkit import Chem
from rdkit.Chem import AllChem


def save_xyz_file(path, one_hot, positions, node_mask, names, suffix=''):
    idx2atom = const.IDX2ATOM

    for batch_i in range(one_hot.size(0)):
        mask = node_mask[batch_i].squeeze()
        n_atoms = mask.sum()
        atom_idx = torch.where(mask)[0]

        f = open(os.path.join(path, f'{names[batch_i]}_{suffix}.xyz'), "w")
        f.write("%d\n\n" % n_atoms)
        atoms = torch.argmax(one_hot[batch_i], dim=1)
        for atom_i in atom_idx:
            atom = atoms[atom_i].item()
            atom = idx2atom[atom]
            f.write("%s %.9f %.9f %.9f\n" % (
                atom, positions[batch_i, atom_i, 0], positions[batch_i, atom_i, 1], positions[batch_i, atom_i, 2]
            ))
        f.close()

def load_xyz_files(path, suffix=''):
    files = []
    for fname in os.listdir(path):
        if fname.endswith(f'_{suffix}.xyz'):
            files.append(fname)
    files = sorted(files, key=lambda f: -int(f.replace(f'_{suffix}.xyz', '').split('_')[-1]))
    return [os.path.join(path, fname) for fname in files]


def load_molecules_xyz(file):
    """
    Read an XYZ file and convert it to an RDKit molecule without adding bonds.
    Returns only the RDKit molecule.
    """
    mols = []
    # Read the XYZ file
    with open(file, encoding='utf8') as f:
        # while the end of the file is not reached
        while True:
            line = f.readline()
            if line == '':
                break
            elif line.strip() == '':
                continue
            
            n_atoms = int(line)
            f.readline()  # Skip comment line
            # read until the whole line is a number
            atoms = [f.readline() for _ in range(n_atoms)]
    
            # Create a new RDKit molecule
            mol = Chem.RWMol()
            
            # Add atoms to the molecule
            for i, atom in enumerate(atoms):
                atom_data = atom.split()
                atom_type = atom_data[0]
                
                # Add atom to RDKit molecule
                rdkit_atom = Chem.Atom(atom_type)
                mol.AddAtom(rdkit_atom)
            
            # Convert to regular molecule (not editable)
            mol = mol.GetMol()
            
            # Add 3D coordinates to the molecule
            conf = Chem.Conformer(n_atoms)
            for i, atom in enumerate(atoms):
                atom_data = atom.split()
                position = [float(e) for e in atom_data[1:]]
                conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(
                    position[0],
                    position[1],
                    position[2]
                ))
            mol.AddConformer(conf)

            mols.append(mol)
    
    return mols


def draw_sphere(ax, x, y, z, size, color, alpha):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    xs = size * np.outer(np.cos(u), np.sin(v))
    ys = size * np.outer(np.sin(u), np.sin(v)) #* 0.8
    zs = size * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x + xs, y + ys, z + zs, rstride=2, cstride=2, color=color, alpha=alpha)


def plot_molecule(ax, positions, atom_type, alpha, spheres_3d, hex_bg_color, fragment_mask=None):
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    # Hydrogen, Carbon, Nitrogen, Oxygen, Flourine

    idx2atom = const.IDX2ATOM

    colors_dic = np.array(const.COLORS)
    radius_dic = np.array(const.RADII)
    area_dic = 1500 * radius_dic ** 2

    areas = area_dic[atom_type]
    radii = radius_dic[atom_type]
    colors = colors_dic[atom_type]

    if fragment_mask is None:
        fragment_mask = torch.ones(len(x))

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = idx2atom[atom_type[i]], idx2atom[atom_type[j]]
            draw_edge_int = get_bond_order(atom1, atom2, dist)
            line_width = (3 - 2) * 2 * 2
            draw_edge = draw_edge_int > 0
            if draw_edge:
                if draw_edge_int == 4:
                    linewidth_factor = 1.5
                else:
                    linewidth_factor = 1
                linewidth_factor *= 0.5
                ax.plot(
                    [x[i], x[j]], [y[i], y[j]], [z[i], z[j]],
                    linewidth=line_width * linewidth_factor * 2,
                    c=hex_bg_color,
                    alpha=alpha
                )

    # from pdb import set_trace
    # set_trace()

    if spheres_3d:
        # idx = torch.where(fragment_mask[:len(x)] == 0)[0]
        # ax.scatter(
        #     x[idx],
        #     y[idx],
        #     z[idx],
        #     alpha=0.9 * alpha,
        #     edgecolors='#FCBA03',
        #     facecolors='none',
        #     linewidths=2,
        #     s=900
        # )
        for i, j, k, s, c, f in zip(x, y, z, radii, colors, fragment_mask):
            if f == 1:
                alpha = 1.0

            draw_sphere(ax, i.item(), j.item(), k.item(), 0.5 * s, c, alpha)

    else:
        ax.scatter(x, y, z, s=areas, alpha=0.9 * alpha, c=colors)


def plot_data3d(positions, atom_type, camera_elev=0, camera_azim=0, save_path=None, spheres_3d=False,
                bg='black', alpha=1., fragment_mask=None):
    black = (0, 0, 0)
    white = (1, 1, 1)
    hex_bg_color = '#FFFFFF' if bg == 'black' else '#000000' #'#666666'

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(elev=camera_elev, azim=camera_azim)
    if bg == 'black':
        ax.set_facecolor(black)
    else:
        ax.set_facecolor(white)
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False

    if bg == 'black':
        ax.xaxis.line.set_color("black")
    else:
        ax.xaxis.line.set_color("white")

    plot_molecule(
        ax, positions, atom_type, alpha, spheres_3d, hex_bg_color, fragment_mask=fragment_mask
    )

    max_value = positions.abs().max().item()
    axis_lim = min(40, max(max_value / 1.5 + 0.3, 3.2))
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_zlim(-axis_lim, axis_lim)
    dpi = 120 if spheres_3d else 50

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi)
        # plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi, transparent=True)

        if spheres_3d:
            img = imageio.imread(save_path)
            img_brighter = np.clip(img * 1.4, 0, 255).astype('uint8')
            imageio.imsave(save_path, img_brighter)
    else:
        plt.show()
    plt.close()


