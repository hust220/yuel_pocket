#%%

import pickle
import numpy as np
import os

# Path to the pickle files
base_path = "/storage/home/juw1179/scratch/datasets/allorank/asd"
features_path = os.path.join(base_path, "features.pkl")
labels_path = os.path.join(base_path, "labels.pkl")

# Read features
with open(features_path, 'rb') as f:
    features = pickle.load(f)
    print(len(features))
    print([len(f) for f in features])
    print([len(l) for l in features[0]])

# Read labels
with open(labels_path, 'rb') as f:
    labels = pickle.load(f)
    print(len(labels))

    
    # Try to convert to numpy array if possible
    try:
        labels_array = np.array(labels)
        print(f"\nLabels as numpy array:")
        print(f"Shape: {labels_array.shape}")
        print(f"Data type: {labels_array.dtype}")
    except ValueError as e:
        print(f"\nCould not convert labels to numpy array: {str(e)}")

# %%
