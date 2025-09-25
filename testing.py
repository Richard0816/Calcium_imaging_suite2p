"""import numpy as np
from pathlib import Path
import suite2p
output_ops = np.load('D:\\data\\2p_shifted\\2024-06-03_00001\\suite2p\\plane0\\spks.npy', allow_pickle=True)
print(output_ops)

f_reg = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename='D:\\data\\2p_shifted\\2024-06-03_00001\\suite2p\\plane0\\data.bin', n_frames = f_raw.shape[0])
print(f_reg)
#f_cells = np.load(Path(output_ops['D:\\data\\2p_shifted\\2024-06-03_00001\\suite2p\\plane0\\']).joinpath('F.npy'))
"""

"""import psutil

# Get virtual memory statistics
virtual_memory = psutil.virtual_memory()

# Total physical memory
total_memory_gb = round(virtual_memory.total / (1024**3), 2)
print(f"Total Physical Memory: {total_memory_gb} GB")

# Available memory
available_memory_gb = round(virtual_memory.available / (1024**3), 2)
print(f"Available Memory: {available_memory_gb} GB")"""

import numpy as np
import matplotlib.pyplot as plt

#root = 'D:\\data\\2p_shifted\\2024-06-03_00001\\suite2p\\plane0\\'
#root = 'D:\\data\\2p_shifted\\2024-07-01_00019\\suite2p\\plane0\\'
#root = 'D:\\data\\2p_shifted\\2024-06-03_00004\\suite2p\\plane0\\'
root = 'D:\\data\\2p_shifted\\2024-11-05_00007\\suite2p\\plane0\\'

F = np.load(root+'F.npy', allow_pickle=True)
Fneu = np.load(root+'Fneu.npy', allow_pickle=True)
spks = np.load(root+'spks.npy', allow_pickle=True)
stat = np.load(root+'stat.npy', allow_pickle=True)
ops = np.load(root+'ops.npy', allow_pickle=True)
ops = ops.item()
iscell = np.load(root+'iscell.npy', allow_pickle=True)


print(len(F[0]))


def derv_plot(lst):
    final_lst =[]
    for i in range(len(lst)-2):
        final_lst.append(lst[i+2]-lst[i])
    return final_lst


def temp(iscell):
    for i in range(len(iscell)):
        x = iscell[i].tolist()
        if x[0] == 1 and float(x[1]) > 0.9:
            return i
print(temp(iscell))

F_naught = np.mean(Fneu[0:2000])

#n=3
#n=18
#n=2
n=126

plt.plot(Fneu[n]/F_naught)
#plt.plot(F[3])
plt.plot(derv_plot(Fneu[n]/F_naught))
plt.show()

"""import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and will be used.")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU.")"""