# A Friendly Walkthrough of `pipeline_gui.py`

> **Who this is for:** a curious high school student. You've had algebra and a little bit of biology. You've maybe written a Python script. You don't need to know what a Butterworth filter is — we'll explain everything as we go.
>
> **What this document covers:** the whole calcium-imaging app called **CalLIOPE** (Calcium Live-imaging Output Pipeline for Epileptiform-recordings). It has 7 tabs and each tab is a step in a long science assembly line. We'll go tab by tab and explain *what* it does, *why* a neuroscientist needs that step, and *how* the math under the hood actually works.

---

## Table of Contents

1. [What problem is this whole thing solving?](#1-what-problem-is-this-whole-thing-solving)
2. [The big picture: an assembly line for brain videos](#2-the-big-picture-an-assembly-line-for-brain-videos)
3. [How the tabs talk to each other (`AppState`)](#3-how-the-tabs-talk-to-each-other-appstate)
4. [Tab 1 — Input & Preprocess](#4-tab-1--input--preprocess)
5. [Tab 2 — QC Preview](#5-tab-2--qc-preview)
6. [Tab 3 — Suite2p Detection (finding the cells)](#6-tab-3--suite2p-detection-finding-the-cells)
7. [Tab 4 — Low-pass filter (cleaning the signal)](#7-tab-4--low-pass-filter-cleaning-the-signal)
8. [Tab 5 — Event detection (when did the brain "go off"?)](#8-tab-5--event-detection-when-did-the-brain-go-off)
9. [Tab 6 — Clustering (which neurons are friends?)](#9-tab-6--clustering-which-neurons-are-friends)
10. [Tab 7 — Cross-correlation (who leads, who follows?)](#10-tab-7--cross-correlation-who-leads-who-follows)
11. [Glossary of every weird word in this document](#11-glossary)

---

## 1. What problem is this whole thing solving?

Neurons in the brain "talk" by firing tiny electrical pulses called **action potentials**. You can't see electricity with a microscope, so neuroscientists do a clever trick: they put a glowing chemical inside neurons that lights up when calcium ions flow in. Calcium flows in *whenever a neuron fires*. So if you point a microscope at a slice of brain that's been treated this way, you see flashing dots — every flash is a neuron firing.

The microscope records a **video**. The video might be 30 minutes long, with hundreds of neurons in view, each flashing dozens of times. The raw video is just brightness numbers — a giant 3D pile of pixels (width × height × frames). Hidden inside that pile of numbers are scientific questions like:

- *"Where are the cells?"*
- *"When did each cell fire?"*
- *"Did groups of cells fire together?"*
- *"Did cell A fire before cell B every time?"*

This software, **CalLIOPE**, turns the raw video into answers to those questions, one click at a time. Every tab handles one part of that translation.

---

## 2. The big picture: an assembly line for brain videos

Think of it like a factory:

| Tab | What goes in | What comes out |
|-----|--------------|----------------|
| 1. Preprocess | Raw TIFF video from the microscope | Cleaned-up video + a tiny preview GIF + first guess at where cells are |
| 2. QC Preview | The GIF from Tab 1 | (Just a sanity check — *did Tab 1 work?*) |
| 3. Suite2p Detection | Cleaned video | Final list of neurons + brightness traces over time |
| 4. Low-pass Filter | Brightness traces | Smoothed traces (less noise) |
| 5. Event Detection | Smoothed traces | Times when **lots** of neurons fired together |
| 6. Clustering | Brightness traces | Groups of similar-acting neurons |
| 7. Cross-correlation | Two groups of neurons | Time-delay numbers showing who leads whom |

Each tab depends on the one before it. You can't cluster cells until you've found cells. You can't find cells until you've cleaned the video. So the user is basically pressing "Run" on each tab in order, top to bottom.

---

## 3. How the tabs talk to each other (`AppState`)

Open [pipeline_gui.py:56](pipeline_gui.py#L56) and you'll see a class called `AppState`. Every tab gets a reference to *the same* `AppState` object. Whenever one tab finishes, it sets a value on `AppState`, and `AppState` calls every other tab's listener function so they know "hey, new data is ready."

It's like a group chat: tabs don't text each other directly. They post in the chat ("I'm done!") and anyone who cares gets the notification.

The three things being broadcast are:

- `result` — the preprocessing finished (Tab 1 → everyone)
- `plane0` — Suite2p found the cells (Tab 3 → Tabs 4, 5, 6, 7)
- `lowpass_plane0` — the smoothed traces are saved (Tab 4 → Tab 5)

Why bother with this design? Because it means each tab is a self-contained unit. If you want to add an 8th tab, you don't have to touch Tabs 1–7 — you just write the new tab and have it `subscribe()` to whatever event it cares about.

---

## 4. Tab 1 — Input & Preprocess

**Defined in [pipeline_gui.py:109](pipeline_gui.py#L109) (`PreprocessTab`).**

### Why this tab exists

The TIFF file from the microscope is *raw*. It has three problems:

1. **Wonky pixel values.** Some frames have negative numbers (instrument noise) or values that are too big to fit in the standard image format. The image needs to be rescaled.
2. **It's huge.** A 30-minute video at 15 frames/second with 2 megapixel frames is many gigabytes. You can't watch it back to check it.
3. **You haven't even *seen* it yet.** You want to eyeball it for 10 seconds to make sure the experiment didn't fail (camera unplugged, brain dried out, etc.).

This tab fixes all three problems before any heavy science begins.

### What the user does

- **Browse** to find a folder containing one or more `.tif` files.
- **Pick** one (or several) to combine into one recording.
- Optionally click **Advanced** to tweak parameters — the defaults are good but the dialog lets you change things like the expected size of a cell body in pixels.
- Click **Run**.
- Watch a live log scroll by (powered by a queue, see [pipeline_gui.py:810](pipeline_gui.py#L810)) telling you what step is happening.

### What happens under the hood — `preprocessing.py`

When you click Run, the GUI calls `preprocessing.preprocess_tiff()` on a background thread (so the GUI doesn't freeze). That function does three jobs:

#### Job 1: "Shifting" the pixel values

The microscope sometimes spits out negative numbers. Image formats like `uint16` (16-bit unsigned integer) only allow whole numbers from 0 to 65,535 — no negatives. So the function finds the most negative value across the whole video and *adds that much* to every pixel. Now the smallest pixel is 0 and everything is positive.

```
shifted_pixel = original_pixel + |smallest_value|
```

That's literally the math. You're just sliding the whole brightness axis up.

#### Job 2: Building the "mean image"

Imagine taking every frame of the video and stacking them on top of each other, then averaging the brightness at each pixel. That's the **mean image**. It's a single 2D photo that shows the *typical* brightness pattern across the recording. Cells that fire often will appear bright; the empty space between cells will be dark.

It's literally: `mean[y, x] = sum_over_frames(video[t, y, x]) / number_of_frames`

#### Job 3: Detecting "blobs" (preview cells)

A "blob" is just a roughly circular bright spot. Cells look like blobs because they're little blob-shaped objects. The detector uses an algorithm called **Laplacian of Gaussian** (LoG). Don't worry about the name. Here's the intuition:

- Take the mean image.
- Look at every pixel. Compare its brightness to the *average brightness of pixels at a certain distance around it*.
- If the center is much brighter than the surroundings, this pixel is likely the middle of a blob.
- The "Gaussian" part means the comparison is weighted — pixels right next to the center matter more than pixels far away.
- You repeat this at several "blob sizes" (e.g. small blob, medium blob, big blob) so you can find blobs of different cell sizes.

This is a *first guess*. It's not the final list of cells — Tab 3 does that more carefully. But it's enough to glance at and confirm "yes, those bright dots look like cells."

#### Job 4: Saving a tiny preview GIF

The full video is too big to scrub through. So the tab makes a downsampled animated GIF — fewer frames, smaller resolution, with the blob circles drawn on top. That's what Tab 2 will play.

### What gets saved to disk

In a folder named after the recording:

- `shifted_tiff.npy` — the pixel-shifted video as a NumPy array
- `mean.npy` — the mean image
- `blobs.npy` — an `(N, 3)` array where each row is `(y, x, radius)` of a detected blob
- `qc.gif` — the preview animation

Once the file `qc.gif` exists, `AppState.set_result()` is called and Tab 2 wakes up.

---

## 5. Tab 2 — QC Preview

**Defined in [pipeline_gui.py:493](pipeline_gui.py#L493) (`QcTab`). QC = "Quality Control".**

### Why this tab exists

You just ran a heavy preprocessing step. Before spending another 20 minutes on cell detection, you want to make sure nothing went wrong. Tab 2 is the eyeball test.

### What the user does

Nothing, mostly. The tab automatically loads the `qc.gif` from Tab 1 and starts playing it on a loop on the left side. On the right side, it shows the mean image with little circles drawn around every detected blob.

There's also a **Reload from folder** button so you can come back later, point at a folder, and review a recording you processed yesterday.

### What it shows

- **Left panel:** The animated GIF, frames advancing every ~66 ms (about 15 FPS playback).
- **Right panel:** A static plot — the mean image as the background, with blob circles drawn on top. If the circles line up with the bright spots, your blob detector is working. If half the circles are in empty black space, your detector parameters are wrong and you need to go back to Tab 1's Advanced settings.

The PIL library (`from PIL import Image, ImageDraw`) is what loads the GIF frames. They get converted to `PhotoImage` objects (a tkinter format) and cycled through with a `self.after(66, self._advance_gif)` scheduler — the GUI's way of saying "call this function again in 66 milliseconds."

### What gets saved

Nothing! This tab is read-only. It's just a visual check.

---

## 6. Tab 3 — Suite2p Detection (finding the cells)

**Defined in [pipeline_gui.py:830](pipeline_gui.py#L830) (`Suite2pTab`).**

This is the most computationally heavy tab. It does the *real* cell-finding and produces the brightness traces that the rest of the pipeline depends on.

### Why this tab exists

Tab 1 found *blobs* — circular bright spots on the mean image. But that's not enough. Two cells right next to each other might blur into one blob. A bright artifact (dust, a blood vessel) might look like a blob but isn't a cell. We need a smarter algorithm that uses the *whole video* — not just the mean — to figure out which pixel groups actually behave like neurons (i.e., they get bright, then dim, then bright again, in a pattern consistent with calcium signals).

### What the user does

- Pick an `ops.npy` file (a Suite2p config file with default detection settings).
- Optionally pick a `cell-filter` checkpoint — a small machine-learning model trained to score each ROI's "cell-ness."
- Choose the dF/F baseline mode (more on dF/F below).
- Click **Run detection**.
- Watch the live console — it streams stdout from the detection algorithm in real time using a custom `_QueueWriter` class ([pipeline_gui.py:810](pipeline_gui.py#L810)).
- When done: two panels light up showing the detected ROIs and the filtered ROIs.

### What happens under the hood

There are three big steps. Let's go through them.

#### Step A: Suite2p + Cellpose detection (`sparse_plus_cellpose.run()`)

Two algorithms run in parallel:

- **Sparsery (Suite2p)** — looks at the *time series* of every pixel. If a clump of pixels all get bright and dim together (correlated activity), that clump is probably a cell. Sparsery iteratively peels off the highest-correlated pixel groups and calls each one a region of interest (ROI).
- **Cellpose** — a deep neural network (think: a model that learned what cells look like by seeing thousands of labeled images). It takes the mean image and outlines anything that looks like a cell *based on shape*, not activity.

These two methods are complementary. Sparsery finds active cells but might miss quiet ones. Cellpose finds shape-y cells but might hallucinate things that don't actually fire. So the code **merges** them:

> Keep all Sparsery ROIs. For Cellpose ROIs, keep them only if they don't overlap >30% with a Sparsery ROI.

That gives you the union of both approaches without duplicates.

#### Step B: Computing dF/F (the "language" of calcium signals)

Once you have ROIs, you average the brightness of all pixels inside each ROI for each frame. That gives you `F(t)` — fluorescence over time — for every neuron. But raw `F(t)` is misleading:

- The **dye fades** over time (photobleaching) — every neuron looks dimmer at the end of the recording for reasons that have nothing to do with firing.
- Different neurons have different baseline brightnesses just because of where they sit relative to the laser.

So we don't use raw F(t). We use **dF/F** (read "delta-F over F"), defined as:

```
dF/F(t) = ( F(t) − F₀(t) ) / F₀(t)
```

where `F₀(t)` is the *baseline* — the brightness the cell has when it's *not* firing.

A `dF/F` value of `0.3` means "this cell is currently 30% brighter than its baseline" — which is meaningful no matter how dim or bright the cell normally is.

How is `F₀(t)` estimated? Two options in the GUI:

1. **Rolling baseline (default).** For each time `t`, look at the brightness in a 45-second window around `t` and take the **10th percentile**. Why the 10th percentile? Because the 10th percentile is "the quiet floor" — it's lower than 90% of the values, so it's what the cell looks like when it's *not* in the middle of a spike. The mean would be biased upward by the spikes themselves; the 10th percentile is robust.
2. **First N minutes.** Just pick the median brightness in the first N minutes (when the experiment is presumably "at rest") and use that as the baseline. Simpler but less accurate when the dye fades.

The function that does this is `utils.robust_df_over_f_1d()` (called once per neuron, or in a single big GPU batch via `analyze_output_gpu` if your machine has a GPU).

After this step, every cell has a 1D trace of `dF/F` values, one per frame. These are saved as memory-mapped arrays (`r0p7_dff.memmap`) so we don't have to load gigabytes into RAM every time we want to look at them.

#### Step C: Cell filtering

Even after Suite2p+Cellpose, some "ROIs" are junk — dim spots that don't look like real cells. The `cell-filter` model is a neural network trained to score each ROI between 0 and 1 ("how cell-like is this?"). The tab saves:

- `predicted_cell_mask.npy` — a boolean array, `True` if the ROI passes the filter
- `predicted_cell_prob.npy` — the actual scores, useful for ranking

The "filtered dF/F" memmap (`r0p7_filtered_dff.memmap`) only includes the cells that passed.

### What gets shown

- **Live console** (left, bottom) — print statements from the detection in real time
- **Panel 2:** All detected ROIs overlaid as colored shapes on a background image (you can pick the background — mean image, max projection, etc.)
- **Panel 3:** Same view, but only ROIs that passed the cell filter, colored by their predicted-cell probability

### What gets saved

- `F.npy`, `Fneu.npy` — raw fluorescence and neuropil (background) per ROI
- `stat.npy`, `ops.npy`, `iscell.npy` — Suite2p's metadata
- `predicted_cell_mask.npy`, `predicted_cell_prob.npy` — cell-filter outputs
- `r0p7_dff.memmap` — raw dF/F for all ROIs `(time × ROI)`
- `r0p7_dff_lowpass.memmap` — lightly smoothed dF/F (default 1 Hz cutoff)
- `r0p7_dff_dt.memmap` — first derivative of the smoothed signal (used in Tab 5)
- `r0p7_filtered_dff.memmap` — dF/F for only the passing ROIs

When the filtered memmap exists, `AppState.set_plane0()` fires and Tabs 4–7 wake up.

---

## 7. Tab 4 — Low-pass filter (cleaning the signal)

**Defined in [pipeline_gui.py:1761](pipeline_gui.py#L1761) (`LowpassTab`).**

### Why this tab exists

The dF/F traces from Tab 3 are **noisy**. They wiggle frame-by-frame from random sensor noise. Real calcium events happen on the timescale of *seconds*, not milliseconds, so we want to throw away high-frequency wiggles and keep the slow-changing signal. That's what a low-pass filter does.

But — what counts as "high frequency"? Should we throw away anything faster than 0.5 Hz? 1 Hz? 5 Hz? The answer depends on the recording. So this tab gives the user a **slider** to adjust the cutoff frequency (between 0.01 Hz and 10 Hz, default 1 Hz) and shows them, *live*, what the filtered trace looks like.

### Background: what's a "frequency" in a calcium trace?

If a cell flashes once every 2 seconds, that's a frequency of 0.5 Hz (Hz = "per second"). If it flashes 10 times per second, that's 10 Hz. Random sensor noise is mostly *very high* frequency — it changes from frame to frame, which at 15 frames/second is 7+ Hz wiggling.

So the strategy is: keep the slow stuff (the real biology, ~0.1–2 Hz), throw away the fast stuff (>2 Hz, which is noise).

### What the user does

- **The slider** at the top picks a cutoff frequency, in Hz. Drag it and *all three plots update live*.
- **Source radio buttons:** pick which trace to preview — the *mean* across all kept ROIs, the *best-scoring single ROI*, or a *manually-typed ROI #*.
- **Compute button:** when satisfied with the cutoff, click this to write final smoothed memmaps to disk (saves them at the chosen cutoff, used by Tab 5).

### Three plots

1. **FFT spectrum** (top) — a graph of "how much of the signal lives at each frequency?" Spikes show up as humps at low frequencies (0.1–2 Hz). Noise is a flat plateau at high frequencies. A red dashed vertical line marks where your cutoff is. Anything to the *right* of the line gets thrown away.
2. **Raw trace** (middle) — the original noisy dF/F.
3. **Filtered trace** (bottom) — the smoothed result. The smoother it is, the higher you've set the cutoff (more aggressive smoothing).

### How the filter works — `utils.lowpass_causal_1d()`

The filter is a **Butterworth filter**, a classic signal-processing tool. The recipe:

- Decide a cutoff frequency `f_c` in Hz.
- Build a mathematical formula that *passes* frequencies below `f_c` (multiplies them by ~1) and *attenuates* frequencies above `f_c` (multiplies them by very small numbers).
- Run the input signal through this formula sample by sample.

The "causal" part is important: the filter only uses *past and present* samples, never future ones. This means it doesn't shift events in time — if a real spike happens at second 10, the filtered trace also shows it at second 10. Non-causal (zero-phase) filters are smoother but they "smear" spikes both forward and backward in time, making it look like the cell started firing before it actually did. For event detection in Tab 5, that would be a disaster.

### Why also save a "derivative"?

Tab 5 needs to detect *when a cell starts firing*. The raw dF/F goes up slowly during a spike — there's no sharp moment. But the **derivative** (rate of change, `d(dF/F)/dt`) is high at the moment a spike starts and low when the cell is steady. So the derivative is what gets thresholded for spike onsets.

It's computed with a **Savitzky–Golay** filter — a special kind of smoothing that simultaneously denoises and computes the derivative. Defined in `utils.sg_first_derivative_1d()`.

### What gets saved

- `r0p7_filtered_dff_lowpass.memmap` — filtered dF/F at the chosen cutoff (only the passing ROIs)
- `r0p7_filtered_dff_dt.memmap` — derivative of the above

When these exist, `AppState.set_lowpass_ready()` fires and Tab 5 unlocks.

---

## 8. Tab 5 — Event detection (when did the brain "go off"?)

**Defined in [pipeline_gui.py:2324](pipeline_gui.py#L2324) (`EventDetectionTab`).**

### Why this tab exists

Sometimes lots of neurons fire at once — that's called a **population event**. In epilepsy research, these can correspond to seizure-like events. We want to know:

- **When** did each population event happen? (Start time, end time.)
- **Which** neurons participated in each event?
- **Who fired first** within each event?

This tab answers all three questions.

### What the user does

- Click **Render**. (Optionally click **Advanced** first to tweak parameters — there are 23 of them, grouped by stage.)
- After rendering, click **Save summary** to write the results to an Excel spreadsheet.

### Step-by-step under the hood

This is a multi-stage algorithm. Bear with me — each step is intuitive on its own.

#### Stage 1: Per-ROI spike onset detection (`utils.hysteresis_onsets`)

For each cell:

1. Take its derivative trace (the `r0p7_filtered_dff_dt` memmap from Tab 4).
2. Compute a **robust z-score** using `utils.mad_z()`: instead of dividing by the standard deviation (which is biased by spikes themselves), we divide by `1.4826 × MAD`, where MAD is the **median absolute deviation**. For normal data this gives the same answer as the standard deviation, but for spiky data the MAD ignores the spikes and gives a clean estimate of the background noise level. So `z = (x − median(x)) / (1.4826 × MAD(x))`.
3. Apply **hysteresis thresholding**: a spike "starts" when `z` goes above an upper threshold (e.g. 3.5) and "ends" only when it falls below a lower threshold (e.g. 1.5). The two-threshold trick prevents one spike from being counted as several when the signal wiggles around the threshold.

The output is a list of spike-start times for each cell.

#### Stage 2: Population density

Now stack all the spikes from all cells onto a single timeline and bin them. For example, in 0.5-second bins: how many spike-starts happened between second 0 and 0.5? Between 0.5 and 1.0? Etc. The result is a 1D histogram — the **population density** of activity over time.

Smooth that histogram with a Gaussian filter (a little blur) so noisy bins don't trigger false events.

#### Stage 3: Peak detection

Use `scipy.signal.find_peaks` to locate maxima in the smoothed density. A "peak" only counts if it sticks up at least `min_prominence` above the surrounding baseline — this prevents tiny bumps from being labeled as events.

#### Stage 4: Boundary walking

Now we know *where* the events peaked. But where do they *start* and *end*? For each peak:

1. Compute a baseline level — the rolling 5th percentile of density nearby — call it `b`.
2. Compute a noise level — the median absolute deviation of the density nearby — call it `n`.
3. Walk leftward from the peak until the density drops below `b + k×n`. That's the start.
4. Walk rightward from the peak until it drops below the same threshold. That's the end.

The constant `k` is tunable (default ~2). A higher `k` means tighter, more conservative boundaries.

#### Stage 5 (optional): Gaussian refinement

Some events are bell-shaped. The algorithm can optionally fit a Gaussian curve to each peak and use the curve's width as the boundary instead. Whichever method (boundary-walk or Gaussian) gives the *tighter* window is used.

#### Stage 6: Per-event participation

For each event window `[start, end]` and each cell, check: did this cell have any spike-start inside this window? If yes, mark it as participating, and record the *time* of its first spike. The output is two big matrices:

- `A[cell, event]` = `True/False` (did cell participate in event?)
- `first_time[cell, event]` = seconds (when did cell first fire in this event? `NaN` if it didn't)

These let you ask "in event #5, who fired first?" by looking at the column of `first_time`.

### What gets shown

- **Heatmap (top):** every row is a cell, every column is time, color is dF/F brightness. Rows are sorted by how active each cell is. Bright vertical bands = population events.
- **Raster (middle):** same axes, but each pixel is on/off — black if a spike-start happened there, white otherwise. Vertical alignments of spikes = population events.
- **Diagnostics (bottom):** the smoothed population density curve, with detected peaks marked, baseline drawn, and event windows shaded. This is the output of `utils.plot_event_detection()` and `utils.shade_event_windows()`.

### What gets saved

The Save Summary button writes Excel sheets via `summary_writer.write_events_sheets()`:

- `EventWindows` sheet: one row per event with `start_s`, `peak_s`, `end_s`.
- `EventOnsets` sheet: one row per (event, cell) participation, including the first-spike time.

Tab 7 (cross-correlation) uses these for its "per-event mode."

---

## 9. Tab 6 — Clustering (which neurons are friends?)

**Defined in [clustering_tab.py](clustering_tab.py).**

### Why this tab exists

100 neurons firing isn't just 100 independent cells — they often form *teams*. Some teams fire together during one type of event; other teams fire together during another. Clustering finds these teams *automatically*.

### What "clustering" means

Clustering is the act of grouping things by similarity. If you have 100 dots on a graph, clustering tells you "those 30 dots in the upper left form one group, those 40 in the middle form another, and the remaining 30 are scattered." For neurons, "similarity" means **their dF/F traces look similar over time**.

We use **hierarchical clustering**: it builds a tree (called a **dendrogram**) where the most similar neurons are joined first, then progressively merged into bigger and bigger groups, until at the very top everyone is one big group. The user "cuts" this tree at some height to decide how many clusters they want.

### What the user does

- Pick the recording's `plane0` folder.
- Click **Run analysis**.
- Look at the dendrogram and spatial map.
- Optionally drag a slider to manually pick where the tree gets cut.
- Pick a color palette (categorical = each cluster gets a distinct color; continuous = colors form a gradient).
- Click **Export** to save each cluster as a `.npy` file for downstream analysis.

### Under the hood

#### Step 1: Compute pairwise similarity

For every pair of neurons `(i, j)`, compute the **Pearson correlation** of their dF/F traces. Pearson is a number between `−1` and `+1`:
- `+1` = the two traces rise and fall in perfect lockstep.
- `0` = totally unrelated.
- `−1` = one rises when the other falls.

Convert this to a *distance*: `d(i, j) = 1 − correlation`. Now traces that are very similar have small distance; unrelated traces have distance ~1.

#### Step 2: Build the tree (`scipy.cluster.hierarchy.linkage`)

Repeatedly merge the two closest groups until everyone's in one group. The order of merges is recorded in a **linkage matrix** that encodes the dendrogram.

#### Step 3: Cut the tree (`scipy.cluster.hierarchy.fcluster`)

Pick a height threshold. Every branch *below* that threshold becomes its own cluster. Drag the slider higher → fewer, bigger clusters. Drag lower → more, smaller clusters.

### What gets shown

- **Dendrogram** (left): a tree diagram. Y-axis = distance at which each merge happened. The dashed horizontal line is the cut. Branches below the cut share a color.
- **Spatial map** (right): the actual position of each cell in the brain, colored by which cluster it belongs to. *This is the cool one* — if cells in cluster A are physically near each other, that's a hint that they're an anatomical or functional unit.

### What gets saved

- One `.npy` file per cluster, listing the ROI indices in that cluster.
- An entry in `calliope_summary.xlsx` (the summary spreadsheet) listing each ROI's cluster assignment.

---

## 10. Tab 7 — Cross-correlation (who leads, who follows?)

**Defined in [crosscorrelation_tab.py](crosscorrelation_tab.py).**

### Why this tab exists

Two neurons can be perfectly correlated — every time A fires, B fires too — but if A *always fires 200 ms before B*, that's a giant clue. It might mean A is sending signals to B (or both share a common source that reaches A first). Cross-correlation measures this **time delay between two signals**.

### Pearson cross-correlation in plain English

Take signal A and signal B, both 30 minutes long. Slide A in time relative to B by an amount we call the **lag**. At each lag, compute Pearson correlation between the two:

- Lag = 0: they line up perfectly in time. What's their correlation?
- Lag = +0.5 s: shift A *forward* by half a second. Now what's the correlation?
- Lag = −0.5 s: shift A *backward*. Now what?

Repeat for every lag in some range, e.g. −2s to +2s. Plot the result and you get a curve called the **cross-correlation function**. The location of the **peak** tells you the time delay where A and B match best.

If the peak is at +0.3 s, that means *A leads B by 0.3 s* (A's signal looks most like B's signal when you shift A forward by that much, i.e. when you let B catch up). If the peak is at 0, they're synchronous.

### The clever speedup — `crosscorrelation.batch_xcorr_clusters`

The naive approach: for every pair `(i, j)` of neurons, run scipy's `correlate` function. With 200 neurons, that's 200 × 200 = 40,000 pairs, each taking real time. Slow.

The trick (in `crosscorrelation.py`):

1. **Z-score every neuron's trace once** (subtract mean, divide by std). This means computing Pearson r reduces to a plain dot product.
2. For each lag `k` in your search range, do **one matrix multiplication** that gives you the correlation of every cell-A vs. every cell-B at that lag.
3. Track the running max across lags so you end up with `(best_lag, max_corr, zero_lag_corr)` matrices.

A matrix multiply runs at the speed of optimized BLAS / cuBLAS — that's the same code that powers deep learning. So this is ~99× faster than per-pair scipy.

### What the user does

This tab has *three modes*:

#### Mode A — Full-recording cross-correlation

Use the entire 30-minute recording. Best for "what's the typical wiring of these clusters?" Outputs one CSV with one row per cell pair.

#### Mode B — Per-event cross-correlation

Re-runs the same analysis but only inside the event windows from Tab 5. So you get separate timing estimates for each population event. Useful when wiring might change across events (e.g. early vs. late in a seizure).

#### Mode C — Single-pair preview

Pick two specific cells, plot the full cross-correlation curve in an embedded matplotlib panel. Used as a sanity check or to dive deep into one interesting pair.

The plot shows:
- X-axis: lag in seconds
- Y-axis: Pearson r at that lag
- A red dashed vertical line at the peak (best lag)
- A green dot at lag = 0 (the zero-lag correlation value)

### What gets saved

A CSV with columns:

| column | meaning |
|--------|---------|
| `cluster_pair` | which two clusters were compared |
| `roi_A`, `roi_B` | indices of the specific cells |
| `best_lag_sec` | the lag where correlation peaks (positive = A leads B) |
| `max_corr` | Pearson r at the peak (−1 to +1) |
| `zero_lag_corr` | Pearson r at lag 0 |
| (other) | fps, n_frames, event window index for per-event mode |

This CSV is the final scientific output of the pipeline. From here, the researcher uses statistics tools (R, Python notebooks, Excel) to ask questions like *"Is the lag from cluster X to cluster Y consistent across events? Does it change in epileptic recordings?"*

---

## 11. Glossary

- **Action potential** — a brief electrical spike that a neuron generates when it "fires."
- **Baseline (F₀)** — the brightness of a cell when it isn't actively firing. Used as the reference point for dF/F.
- **Blob** — a roughly circular bright spot. A first guess at a cell.
- **Butterworth filter** — a common type of frequency filter, designed to be maximally flat in the "pass" band.
- **Calcium imaging** — recording fluorescence from a calcium-sensitive dye/protein to indirectly observe neural firing.
- **Cellpose** — a deep-learning tool that segments cells based on their shape.
- **Cluster** — a group of items (here, neurons) that are similar to each other.
- **Cross-correlation** — measuring how similar two signals are at every possible time-shift.
- **Cutoff frequency** — the frequency above which a low-pass filter starts attenuating signal.
- **dF/F (delta-F over F)** — `(F − F₀) / F₀`, a baseline-corrected measure of how much brighter a cell is than its resting brightness.
- **Dendrogram** — a tree diagram showing hierarchical clustering results.
- **Derivative** — the rate of change of a signal. Big where signal is rising fast, near zero where it's flat.
- **FFT (Fast Fourier Transform)** — an algorithm that decomposes a signal into the strengths of all its frequency components.
- **Frame** — one image in a video. A 30-min recording at 15 FPS has ~27,000 frames.
- **GIF** — a compressed animated image format, used here for the QC preview.
- **Hierarchical clustering** — a clustering method that builds a tree of progressively merged groups.
- **Hysteresis** — using two thresholds (high to enter, low to exit) instead of one, so noise doesn't cause flickering.
- **Hz (hertz)** — cycles per second. 1 Hz = once per second.
- **Lag** — a time-shift. "A leads B by 0.3 s" = lag of +0.3 s.
- **Laplacian of Gaussian (LoG)** — a blob detector that finds spots brighter than their surroundings.
- **Low-pass filter** — keeps low-frequency content, removes high-frequency content.
- **MAD (Median Absolute Deviation)** — a robust measure of spread: the median of `|x − median(x)|`. Robust to outliers, unlike standard deviation.
- **Memmap (memory-mapped array)** — a NumPy array stored on disk but accessed like in-memory data. Used for large traces that don't fit in RAM.
- **Pearson correlation** — a number from −1 to +1 measuring linear similarity between two signals.
- **Population event** — a moment when many neurons fire at once.
- **ROI (Region of Interest)** — a set of pixels representing one cell.
- **Robust z-score** — z-score using MAD instead of standard deviation.
- **Savitzky–Golay filter** — a smoothing filter that can also compute derivatives.
- **Sparsery** — an algorithm in Suite2p that finds active cells from temporal correlation.
- **Suite2p** — a popular Python pipeline for processing calcium imaging data.
- **TIFF** — a common image format used by microscopes; supports stacks of frames.
- **Z-score** — `(x − mean) / standard_deviation`. Tells you how many standard deviations from the mean a value is.

---

## A final thought

The whole point of this app is to take *raw pixels* and turn them into *answers about neurons*. Every tab handles one specific transformation:

> **Pixels → cleaned video → cells → traces → smoothed traces → events → groups → timing relationships**

If you ever come back to this code and feel lost, just remember: each tab is one arrow in that chain. Find the tab, read what it produces, and you'll know where you are.
