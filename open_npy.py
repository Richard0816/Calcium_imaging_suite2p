import numpy as np

path_to_ops = r'D:\data\2p_shifted\2024-06-03_00004\suite2p\plane0\ops.npy'
ops = np.load(path_to_ops, allow_pickle=True).item()

print(ops)