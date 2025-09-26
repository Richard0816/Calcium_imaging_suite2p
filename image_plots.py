import os
import numpy as np
import matplotlib.pyplot as plt

# --- Config ---
root = 'D:\\data\\2p_shifted\\2024-06-03_00002\\suite2p\\plane0\\'
fps = 30.0
roi = 1
t_max = None        # seconds to plot (None for full trace)
z_enter = 3.5     # enter threshold (MAD-z)
z_exit  = 2     # exit threshold (MAD-z), for hysteresis

# --- Load raw (for panel 1) ---
F = np.load(os.path.join(root, 'F.npy'), allow_pickle=True)
Fneu = np.load(os.path.join(root, 'Fneu.npy'), allow_pickle=True)
if F.shape[0] < F.shape[1]:
    raw_trace = F[roi, :]
    neu_trace = Fneu[roi, :]
    nROIs, T = F.shape
else:
    raw_trace = F[:, roi]
    neu_trace = Fneu[:, roi]
    T, nROIs = F.shape

# --- Load processed memmaps (time-major T x N) ---
dff = np.memmap(os.path.join(root, 'r0p7_dff.memmap.float32'), dtype='float32', mode='r', shape=(T, nROIs))
low = np.memmap(os.path.join(root, 'r0p7_dff_lowpass.memmap.float32'), dtype='float32', mode='r', shape=(T, nROIs))
dt  = np.memmap(os.path.join(root, 'r0p7_dff_dt.memmap.float32'), dtype='float32', mode='r', shape=(T, nROIs))

dff_trace = np.asarray(dff[:, roi])
low_trace = np.asarray(low[:, roi])
dt_trace  = np.asarray(dt[:, roi])

# --- Robust z-score (MAD) of derivative ---
med = np.median(dt_trace)
mad = np.median(np.abs(dt_trace - med)) + 1e-12
z = (dt_trace - med) / (1.4826 * mad)

# --- Hysteresis event detection on z ---
above_enter = z >= z_enter
events_onsets = []
active = False
for i in range(len(z)):
    if not active and above_enter[i]:
        active = True
        events_onsets.append(i)
    if active and z[i] <= z_exit:
        active = False

event_times = np.array(events_onsets) / fps

# --- Time axis + crop ---
time = np.arange(T) / fps
if t_max is not None:
    idx = time < t_max
else:
    idx = slice(None)

# --- Plot ---
plt.figure(figsize=(12, 9))

plt.subplot(4, 1, 1)
plt.plot(time[idx], raw_trace[idx], label='F raw')
plt.plot(time[idx], neu_trace[idx], label='F neuropil', alpha=0.7)
plt.legend(loc='upper right'); plt.ylabel('Fluorescence')

plt.subplot(4, 1, 2)
plt.plot(time[idx], dff_trace[idx], label='ΔF/F', color='black')
plt.legend(loc='upper right'); plt.ylabel('ΔF/F')

plt.subplot(4, 1, 3)
plt.plot(time[idx], low_trace[idx], label='Low-pass ΔF/F', color='green')
plt.legend(loc='upper right'); plt.ylabel('Filtered')

plt.subplot(4, 1, 4)
plt.plot(time[idx], dt_trace[idx], label='Derivative', color='red')
# Threshold lines (convert back from z to raw d/dt units for display)
thr_enter_val = med + (1.4826 * mad) * z_enter
thr_exit_val  = med + (1.4826 * mad) * z_exit
plt.axhline(thr_enter_val, linestyle='--', linewidth=1, label=f'Enter z={z_enter:.1f}')
plt.axhline(thr_exit_val,  linestyle=':',  linewidth=1, label=f'Exit z={z_exit:.1f}')
# Event markers
for t0 in event_times:
    if t_max is None or t0 < t_max:
        plt.axvline(t0, linestyle='-', linewidth=0.8, alpha=0.6)

plt.legend(loc='upper right'); plt.ylabel('d/dt (a.u./s)')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

print(f"ROI {roi}: {len(event_times)} events")
if len(event_times) > 0:
    # Print first few event times
    preview = ", ".join(f"{t:.2f}s" for t in event_times[:10])
    print("Event times (first 10):", preview)