import os
import numpy as np
import matplotlib.pyplot as plt

# --- Config (user parameters) ---
root = 'D:\\data\\2p_shifted\\2024-11-20_00004\\suite2p\\plane0\\'  # Path to Suite2p outputs
fps = 30.0          # Imaging frame rate (frames per second)
roi = 2             # ROI index to visualize
t_max = None        # Seconds to plot (None = full trace)
z_enter = 3.5       # z-score threshold to detect event onset
z_exit  = 2         # z-score threshold to detect event offset (hysteresis control)

# --- Load raw Suite2p outputs (panel 1 data) ---
F = np.load(os.path.join(root, 'F.npy'), allow_pickle=True)         # Raw fluorescence
Fneu = np.load(os.path.join(root, 'Fneu.npy'), allow_pickle=True)   # Neuropil fluorescence
if F.shape[0] < F.shape[1]:   # Handle orientation of array (ROIs x Time or Time x ROIs)
    raw_trace = F[roi, :]
    neu_trace = Fneu[roi, :]
    nROIs, T = F.shape
else:
    raw_trace = F[:, roi]
    neu_trace = Fneu[:, roi]
    T, nROIs = F.shape

# --- Load processed ΔF/F traces (time-major arrays: T x N) ---
dff = np.memmap(os.path.join(root, 'r0p7_dff.memmap.float32'),
                dtype='float32', mode='r', shape=(T, nROIs))
low = np.memmap(os.path.join(root, 'r0p7_dff_lowpass.memmap.float32'),
                dtype='float32', mode='r', shape=(T, nROIs))
dt  = np.memmap(os.path.join(root, 'r0p7_dff_dt.memmap.float32'),
                dtype='float32', mode='r', shape=(T, nROIs))

# Extract single ROI traces
dff_trace = np.asarray(dff[:, roi])
low_trace = np.asarray(low[:, roi])
dt_trace  = np.asarray(dt[:, roi])

# --- Robust z-score (MAD) of derivative ---
med = np.median(dt_trace)                                  # Median of derivative
mad = np.median(np.abs(dt_trace - med)) + 1e-12            # Median absolute deviation (avoid div by zero)
z = (dt_trace - med) / (1.4826 * mad)                      # Robust z-score (normalizing factor = 1.4826)

# --- Hysteresis event detection based on z-scores ---
above_enter = z >= z_enter
events_onsets = []
active = False
for i in range(len(z)):
    if not active and above_enter[i]:
        # Event starts when crossing enter threshold
        active = True
        events_onsets.append(i)
    if active and z[i] <= z_exit:
        # Event ends when dropping below exit threshold
        active = False

# Convert onset indices to event times in seconds
event_times = np.array(events_onsets) / fps

# --- Time axis and cropping if requested ---
time = np.arange(T) / fps
if t_max is not None:
    idx = time < t_max
else:
    idx = slice(None)

# --- Plot traces ---
plt.figure(figsize=(12, 9))

# Panel 1: Raw + neuropil fluorescence
plt.subplot(4, 1, 1)
plt.plot(time[idx], raw_trace[idx], label='F raw')
plt.plot(time[idx], neu_trace[idx], label='F neuropil', alpha=0.7)
plt.legend(loc='upper right'); plt.ylabel('Fluorescence')

# Panel 2: ΔF/F
plt.subplot(4, 1, 2)
plt.plot(time[idx], dff_trace[idx], label='ΔF/F', color='black')
plt.legend(loc='upper right'); plt.ylabel('ΔF/F')

# Panel 3: Low-pass filtered ΔF/F
plt.subplot(4, 1, 3)
plt.plot(time[idx], low_trace[idx], label='Low-pass ΔF/F', color='green')
plt.legend(loc='upper right'); plt.ylabel('Filtered')

# Panel 4: Derivative trace with thresholds + event markers
plt.subplot(4, 1, 4)
plt.plot(time[idx], dt_trace[idx], label='Derivative', color='red')

# Compute thresholds back in raw derivative units for display
thr_enter_val = med + (1.4826 * mad) * z_enter
thr_exit_val  = med + (1.4826 * mad) * z_exit
plt.axhline(thr_enter_val, linestyle='--', linewidth=1, label=f'Enter z={z_enter:.1f}')
plt.axhline(thr_exit_val,  linestyle=':',  linewidth=1, label=f'Exit z={z_exit:.1f}')

# Add vertical markers at detected event onsets
for t0 in event_times:
    if t_max is None or t0 < t_max:
        plt.axvline(t0, linestyle='-', linewidth=0.8, alpha=0.6)

plt.legend(loc='upper right'); plt.ylabel('d/dt (a.u./s)')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

# --- Print summary ---
print(f"ROI {roi}: {len(event_times)} events")
if len(event_times) > 0:
    preview = ", ".join(f"{t:.2f}s" for t in event_times[:10])
    print("Event times (first 10):", preview)
