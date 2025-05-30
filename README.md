# YuelBond: Bond Order Prediction with Graph Neural Network

## Environment Setup

installing all the necessary packages:

```shell
rdkit
pdb-tools
biopython
imageio
networkx
pytorch
pytorch-lightning
scipy
scikit-learn
tqdm
wandb
```

## Usage

### Preprocess dataset

```shell
mkdir -p models logs trajectories
python preprocess_dataset.py datasets/geom_sanitized_train.pkl datasets/geom_sanitized_train_noise_0_2.pt --noise 0.2
```

### Train

```shell
python -W ignore train_yuel_pocket.py --config configs/train_moad.yml
```

### Design

```shell
wget https://zenodo.org/record/7121300/files/pockets_difflinker_backbone.ckpt?download=1 -O models/pockets_difflinker_backbone.ckpt
python -W ignore yuel_design.py --pocket 2req_pocket.pdb --model test.ckpt --size 15
```





# Reference

> Igashov, I., Stärk, H., Vignac, C. et al. Equivariant 3D-conditional diffusion model for molecular linker design. Nat Mach Intell (2024). https://doi.org/10.1038/s42256-024-00815-9

```
@article{igashov2024equivariant,
  title={Equivariant 3D-conditional diffusion model for molecular linker design},
  author={Igashov, Ilia and St{\"a}rk, Hannes and Vignac, Cl{\'e}ment and Schneuing, Arne and Satorras, Victor Garcia and Frossard, Pascal and Welling, Max and Bronstein, Michael and Correia, Bruno},
  journal={Nature Machine Intelligence},
  pages={1--11},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

# Contact

If you have any questions, please contact me at jianopt@gmail.com
