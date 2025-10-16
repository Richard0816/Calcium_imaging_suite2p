from pathlib import Path
import numpy as np
import pandas as pd

root = r'D:\data\2p_shifted\2024-06-03_00007\suite2p\plane0'
root = Path(root)

is_cell = np.load(root / "iscell.npy", allow_pickle=False)

# --- Inputs ---
csv_path = root / 'criteria.csv'        # path to your CSV
out_path = r'D:\data\2p_shifted\2024-06-03_00007\suite2p\iscell.npy'      # path to save the npy file

# --- Load CSV and extract "label" column ---
df = pd.read_csv(csv_path)
labels = df['label'].values  # first column in output

# --- Second column: 100 if 1, else 0 ---
second_col = np.where(labels == 1, 100, 0)

# --- Stack and save ---
out_arr = np.column_stack((labels, second_col))
np.save(out_path, out_arr)

print(f"Saved {out_arr.shape} array to {out_path}")
