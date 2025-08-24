import pandas as pd
import numpy as np

# File paths
files = [
    'data/train_labels.csv',
    'data/val_labels.csv',
    'data/test_labels.csv'
]

cols_to_randomize = ['durability_score', 'chemical_safety_score', 'ergonomics_score']
all_dfs = []

for file in files:
    df = pd.read_csv(file)
    df_new = df.copy()
    for category in df['category'].unique():
        mask = df['category'] == category
        for col in cols_to_randomize:
            orig_vals = df.loc[mask, col]
            mean = orig_vals.mean()
            std = orig_vals.std() if orig_vals.std() > 0 else 0.5
            # Generate new values close to mean, same count as original
            new_vals = np.random.normal(loc=mean, scale=std*0.2, size=orig_vals.shape[0])
            # Round, clip to 1-10, and cast to int
            new_vals = np.clip(np.round(new_vals), 1, 10).astype(int)
            # Shuffle to avoid order bias
            np.random.shuffle(new_vals)
            df_new.loc[mask, col] = new_vals
    all_dfs.append(df_new)

# Concatenate all and save as single CSV
combined = pd.concat(all_dfs, ignore_index=True)
combined.to_csv('data/all_labels_randomized.csv', index=False)
print('✅ Combined randomized CSV saved: data/all_labels_randomized.csv')
