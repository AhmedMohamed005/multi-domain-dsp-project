# Multi-Domain DSP Project: A to Z Guide

Team size: 6 students. Level: medium. Tools: Python, NumPy, SciPy, Matplotlib, OpenCV.
Domains: mathematical signals, real audio, digital images.

This guide is built from the two files you were given: the project description and the cover sheet. Everything below follows the tasks, simplifications, and assessment rules in those files.

---

## 0. What you must deliver

| Item | Details |
| --- | --- |
| Cover sheet | Fill in the university, faculty, the 6 names and IDs (the "Assigned Part" column is already filled in). |
| Notebook | One `.ipynb` file containing all 6 parts, runnable top to bottom. |
| Filtered audio | `filtered.wav` produced by Part 4. |
| Live discussion | Each student explains their concept, shows code and output, and answers one short question. The team runs the complete project once. |

How you are marked (from the project file): 10 marks per part (6 parts) plus 10 marks for code quality and individual discussion. You are judged on correct implementation and understanding, not on perfect denoising. You do not need a fixed noise reduction like 25 to 30 dB. You report your achieved result and explain limitations.

Important rules from the project file:
- Only the DTFT calculation must be implemented manually (with loops). Everywhere else, use NumPy/SciPy helpers.
- Use `np.convolve` for linear convolution and explain zero-padding.
- The first-order difference equation must be simple.
- The Notch Filter structure is provided by the instructor. You calculate the coefficients for your chosen frequency.
- Use ideal circular masks for the image low-pass and high-pass.
- Group delay and advanced pole-zero analysis are optional.
- The instructor may ask any student to explain a parameter, interpret a plot, or make a small change to their code. Everyone should understand the whole project.

---

## 1. Environment setup

### Option A: Google Colab (recommended for a team)
1. Create one notebook and share the link with all 6 members (Editor access).
2. Upload `audio.wav` and `image.png` using the folder icon on the left.
3. The libraries are already installed.

### Option B: Local Jupyter
```bash
pip install numpy scipy matplotlib opencv-python jupyter
jupyter notebook
```
Put `audio.wav` and `image.png` in the same folder as the notebook.

### The data files
The project says you get a provided WAV file and a provided grayscale image. They were not in the files sent to me, so get them from your instructor. Rename them `audio.wav` and `image.png` (or change the two names in the setup cell). The setup cell below falls back to synthetic data if a file is missing, so you can start coding immediately. Before final submission, run with the real files.

---

## 2. Notebook structure

Use this layout (one master notebook):

1. Title cell (markdown): project name, team, date.
2. Setup cell (code): shared imports, filenames, helper functions.
3. For each Part 1 to 6:
   - markdown cell: title, student name, 3 to 4 lines of theory
   - code cell(s) with comments
   - plots with titles and axis labels
   - markdown cell: what you observed and the limitations
4. Final summary cell: key numbers from all parts.

### Shared setup cell (everyone copies this first)

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import cv2

AUDIO_FILE = "audio.wav"
IMAGE_FILE = "image.png"
np.random.seed(0)          # same noise every run, so results are repeatable

def load_audio(path=AUDIO_FILE):
    """Return (fs, x). x is mono and scaled to [-1, 1]."""
    if os.path.exists(path):
        fs, x = wavfile.read(path)
        x = x.astype(np.float64)
        if x.ndim > 1:                 # stereo -> mono
            x = x.mean(axis=1)
    else:
        print("WAV not found: using a synthetic speech-like signal")
        fs = 8000
        t = np.arange(3 * fs) / fs
        x = (np.sin(2*np.pi*150*t) + 0.6*np.sin(2*np.pi*300*t)
             + 0.4*np.sin(2*np.pi*900*t)) * (0.6 + 0.4*np.sin(2*np.pi*3*t))
    return fs, x / np.max(np.abs(x))

def load_image(path=IMAGE_FILE, size=256):
    """Return a grayscale float image of shape (size, size), values 0..255."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if os.path.exists(path) else None
    if img is None:
        print("Image not found: using a synthetic test image")
        yy, xx = np.mgrid[:size, :size]
        img = 40 + 0.4*xx + 0.2*yy
        img[60:130, 50:140] = 220
        img[((yy-170)**2 + (xx-170)**2) < 40**2] = 120
        img = np.clip(img, 0, 255)
    else:
        img = cv2.resize(img, (size, size))
    return img.astype(np.float64)

def psnr(ref, test):
    """PSNR in dB for 8-bit images. Returns (mse, psnr)."""
    test = np.clip(test, 0, 255)
    mse = np.mean((ref - test) ** 2)
    return mse, (np.inf if mse == 0 else 10*np.log10(255**2 / mse))
```

Why the image is resized to 256 x 256: the size is even, so the centre of the shifted spectrum is exactly at index `size // 2`, and the 2D FFT is fast.

---

## 3. Team plan for 2 days

| When | What |
| --- | --- |
| Day 1 morning | Agree on the shared setup cell, filenames, and sampling rate. Each person starts their part in their own copy of the notebook. |
| Day 1 evening | Each part works and has its plots. |
| Day 2 morning | One person merges all parts into the master notebook. Run Kernel > Restart and Run All. Fix errors. |
| Day 2 afternoon | Each person writes the observation text for their part, fills the cover sheet, and practices their explanation. |
| Before the deadline | Run the whole notebook once more, save with outputs, export a PDF backup, and collect `filtered.wav`. |

Dependencies between parts:
- Part 3 creates the noisy audio that Part 4 filters. In the master notebook Part 4 uses the `noisy` variable from Part 3. To work alone, Part 4 can reload `noisy.wav` (saved by Part 3), or just redo the three lines that create the noise.
- Part 5 and Part 6 both use `load_image()`. They are independent.
- Parts 1 and 2 are independent of everything else.

---

## 4. Part 1: Mathematical signal and DTFT (Student 1)

### Tasks (from the project file)
- Generate a signal with two sinusoidal components and Gaussian noise.
- Compute the DTFT from its defining equation using loops, for a limited number of samples and frequency points.
- Plot the magnitude and phase spectra.

### Theory to explain
Discrete-time signals, angular frequency, DTFT, magnitude and phase.

```
X(ω) = Σ x[n] · e^(−jωn)        n = 0 … N−1
```
- `ω` is angular frequency in radians per sample. A sinusoid with frequency `f` cycles/sample has `ω = 2πf`.
- `X(ω)` is complex. `|X(ω)|` tells how strong each frequency is. `∠X(ω)` tells its time shift.
- A sine at ω0 gives peaks at `+ω0` and `−ω0`.
- `X(ω)` repeats every 2π, so plotting −π to π is enough.
- Using a finite N is like a rectangular window. Each peak has width of about `2π/N`, and two tones can only be separated if `|f1 − f2| > 1/N`.

### Code

```python
# --- Part 1: signal ---
N = 64                                   # short signal (project asks for a limited number of samples)
n = np.arange(N)
f1, f2 = 0.10, 0.25                      # cycles per sample (must be < 0.5)
x = (np.sin(2*np.pi*f1*n) + 0.5*np.sin(2*np.pi*f2*n)
     + 0.3*np.random.randn(N))           # Gaussian noise

# --- DTFT from the definition: two loops, as required ---
K = 512                                  # number of frequency points
w = np.linspace(-np.pi, np.pi, K)
X = np.zeros(K, dtype=complex)
for k in range(K):                       # loop 1: over frequencies
    for i in range(N):                   # loop 2: over samples
        X[k] += x[i] * np.exp(-1j * w[k] * i)

# --- Plots ---
fig, ax = plt.subplots(3, 1, figsize=(9, 9))
ax[0].stem(n, x)
ax[0].set(title="Signal x[n] (two sinusoids + noise)", xlabel="n", ylabel="x[n]")
ax[1].plot(w, np.abs(X))
ax[1].set(title="DTFT magnitude", xlabel="ω (rad/sample)", ylabel="|X(ω)|")
ax[2].plot(w, np.angle(X))
ax[2].set(title="DTFT phase", xlabel="ω (rad/sample)", ylabel="phase (rad)")
for a in ax[1:]:
    for f in (f1, f2):                   # mark the true frequencies
        a.axvline( 2*np.pi*f, color="r", ls="--", alpha=0.5)
        a.axvline(-2*np.pi*f, color="r", ls="--", alpha=0.5)
plt.tight_layout(); plt.show()

# --- Check against FFT (they must agree on the grid ω = 2πk/N) ---
wk = 2*np.pi*np.arange(N)/N
Xk = np.array([sum(x[i]*np.exp(-1j*wk[k]*i) for i in range(N)) for k in range(N)])
print("DTFT matches np.fft.fft:", np.allclose(Xk, np.fft.fft(x)))
print("Expected peaks at ω =", 2*np.pi*f1, "and", 2*np.pi*f2)
```

### What to write in the observations cell
- The two peaks appear at `ω = ±2πf1` and `±2πf2`, matching the true frequencies.
- Noise gives a random floor.
- Phase is erratic where the magnitude is near zero (this is normal).
- The peaks are not infinitely sharp because N is finite.
- Limitation: the double loop is slow, so N and K are kept small.

---

## 5. Part 2: Convolution and difference equations (Student 2)

### Tasks
- Apply a Moving Average filter using linear convolution.
- Compare time-domain convolution with frequency-domain multiplication, using zero-padding to get linear convolution.
- Implement a simple first-order difference equation.

### Theory to explain
Impulse response, LTI systems, convolution, difference equations.

```
y[n] = Σ x[k]·h[n−k]            length(y) = N + M − 1
x * h  <->  X(ω)·H(ω)            (convolution theorem)
```
- An LTI (linear time-invariant) system is completely described by its impulse response `h[n]`. The output for any input is `x * h`.
- The moving average `h = ones(5)/5` smooths the signal, so it is a low-pass filter.
- The FFT assumes a periodic signal, so multiplying FFTs gives circular convolution (the tail wraps around). Zero-padding both signals to length `L >= N+M−1` removes the wrap-around and gives linear convolution.
- Difference equation: `y[n] = a·y[n−1] + x[n]`. Its impulse response is `h[n] = aⁿ` for `n >= 0`. Its transfer function is `H(z) = 1/(1 − a z⁻¹)` with a pole at `z = a`. It is stable if `|a| < 1`.

### Code

```python
# --- Signal ---
N = 50
n = np.arange(N)
x = np.sin(2*np.pi*0.05*n) + 0.4*np.random.randn(N)

# --- 1) Moving average by linear convolution ---
M = 5
h = np.ones(M) / M
y_conv = np.convolve(x, h)                   # 'full' mode: length N+M-1

# --- 2) Same thing through the frequency domain, with zero-padding ---
L = len(x) + len(h) - 1
y_fft = np.real(np.fft.ifft(np.fft.fft(x, L) * np.fft.fft(h, L)))   # fft(.., L) zero-pads
print("Linear conv == FFT method:", np.allclose(y_conv, y_fft),
      "| max error:", np.max(np.abs(y_conv - y_fft)))

# --- Without enough padding: circular convolution (shows why padding matters) ---
y_circ = np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(h, len(x))))
print("Max error without padding (first N samples):",
      np.max(np.abs(y_conv[:N] - y_circ)))

plt.figure(figsize=(9, 7))
plt.subplot(3, 1, 1)
plt.stem(n, x); plt.title("Input x[n]"); plt.xlabel("n")
plt.subplot(3, 1, 2)
plt.plot(y_conv, "o-", label="np.convolve")
plt.plot(y_fft, "x--", label="FFT with zero-padding")
plt.title("Moving average output (the two methods overlap)")
plt.xlabel("n"); plt.legend()
plt.subplot(3, 1, 3)
plt.plot(y_conv[:N] - y_circ)
plt.title("Error of circular convolution (no padding): wrap-around at the start")
plt.xlabel("n")
plt.tight_layout(); plt.show()

# --- Frequency response of the moving average ---
H = np.fft.fft(h, 512)
wgrid = np.linspace(0, 2*np.pi, 512, endpoint=False)
plt.figure(figsize=(7, 3))
plt.plot(wgrid[:256], np.abs(H[:256]))
plt.title("|H(ω)| of the 5-point moving average (low-pass)")
plt.xlabel("ω (rad/sample)"); plt.ylabel("|H|"); plt.show()

# --- 3) First-order difference equation: y[n] = a*y[n-1] + x[n] ---
def first_order(x, a):
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = x[i] + (a * y[i-1] if i > 0 else 0.0)
    return y

a = 0.8
impulse = np.zeros(30); impulse[0] = 1
h_diff = first_order(impulse, a)
print("Impulse response equals a^n:", np.allclose(h_diff, a**np.arange(30)))
print("Matches scipy lfilter:", np.allclose(first_order(x, a),
                                            signal.lfilter([1], [1, -a], x)))

plt.figure(figsize=(9, 5))
plt.subplot(2, 1, 1)
plt.stem(h_diff); plt.title(f"Impulse response of y[n] = {a} y[n-1] + x[n]  (h[n] = {a}^n)")
plt.subplot(2, 1, 2)
plt.plot(x, label="input x[n]"); plt.plot(first_order(x, a), label="output y[n]")
plt.title("Response to the noisy sinusoid (smoothing)"); plt.legend(); plt.xlabel("n")
plt.tight_layout(); plt.show()
```

### Observations to write
- Both convolution methods give the same result (`allclose` is True) only when zero-padded to `N+M−1`.
- Without padding, the beginning of the output is wrong because of circular wrap-around.
- The moving average attenuates high frequencies (see `|H|`), so the noise is reduced and the output is smoother (and slightly delayed).
- For `a = 0.8` the difference equation decays as `0.8ⁿ`. With `|a| >= 1` it would not be stable.

---

## 6. Part 3: Audio analysis (Student 3)

### Tasks
- Load the provided WAV file.
- Add a sinusoidal interference tone at 50 Hz or 60 Hz.
- Plot the clean and noisy waveforms and their frequency spectra, and identify the interference frequency.

### Theory to explain
Sampling frequency, Nyquist theorem, frequency spectrum, noise.

- Sampling: `x[n] = x(n/fs)`.
- Nyquist: to represent frequency `f` you need `fs > 2f`. The highest representable frequency is `fs/2`. Anything above aliases to a wrong lower frequency.
- DFT: `f_k = k·fs/N`, so the frequency resolution is `fs/N` (longer recording, finer resolution).
- For a real signal the spectrum is symmetric, so `rfft` returns only `0 … fs/2`.
- A pure tone appears as one spike in the spectrum even when it is nearly invisible in the waveform. That is how you detect it.
- The 50/60 Hz tone simulates mains hum.

### Code

```python
fs, x = load_audio()                         # clean signal in [-1, 1]
t = np.arange(len(x)) / fs
f_int = 50                                   # interference frequency in Hz (or 60)
A_int = 0.1                                  # interference amplitude
noisy = x + A_int * np.sin(2*np.pi*f_int*t)

print(f"fs = {fs} Hz | duration = {len(x)/fs:.2f} s | Nyquist = {fs/2} Hz")

# --- Spectra ---
def amp_spectrum(sig):
    S = np.abs(np.fft.rfft(sig)) * 2 / len(sig)       # amplitude spectrum
    f = np.fft.rfftfreq(len(sig), 1/fs)
    return f, S

f, S_clean = amp_spectrum(x)
_, S_noisy = amp_spectrum(noisy)

# --- Identify the interference: highest peak in the 40-70 Hz band ---
band = (f >= 40) & (f <= 70)
f_peak = f[band][np.argmax(S_noisy[band])]
print(f"Detected interference frequency: {f_peak:.1f} Hz")

# Cleaner check: the difference of the two spectra shows only the added tone
f_diff = f[np.argmax(np.abs(S_noisy - S_clean))]
print(f"Frequency of largest spectral change: {f_diff:.1f} Hz")

# --- Plots ---
seg = slice(0, int(0.05*fs))                 # first 50 ms, to see the waveform shape
fig, ax = plt.subplots(2, 2, figsize=(11, 7))
ax[0, 0].plot(t[seg], x[seg]);     ax[0, 0].set(title="Clean waveform (first 50 ms)", xlabel="t (s)", ylabel="amplitude")
ax[0, 1].plot(t[seg], noisy[seg]); ax[0, 1].set(title="Noisy waveform (first 50 ms)", xlabel="t (s)", ylabel="amplitude")
ax[1, 0].plot(f, S_clean); ax[1, 0].set(title="Clean spectrum (0-500 Hz)", xlabel="Hz", ylabel="amplitude", xlim=(0, 500))
ax[1, 1].plot(f, S_noisy); ax[1, 1].set(title="Noisy spectrum (0-500 Hz)", xlabel="Hz", ylabel="amplitude", xlim=(0, 500))
ax[1, 1].axvline(f_int, color="r", ls="--", alpha=0.6)
plt.tight_layout(); plt.show()

# --- Save the noisy audio for Part 4 ---
noisy_scaled = noisy / np.max(np.abs(noisy))
wavfile.write("noisy.wav", fs, (noisy_scaled * 32767).astype(np.int16))
```

### Observations to write
- In the waveform the hum is hard to see, but in the spectrum there is a clear spike at 50 Hz.
- State the measured frequency, `fs`, and Nyquist.
- If the tone does not land exactly on an FFT bin, the spike spreads a little to neighbours (spectral leakage).
- Limitation: if the speech itself has strong content near 50 Hz, the peak search band could pick the wrong peak. The spectrum difference check avoids that.

---

## 7. Part 4: Audio filtering with a notch filter (Student 4)

### Tasks
- Apply a second-order Notch Filter to reduce the added interference tone.
- Use the provided filter structure/coefficient equations, and calculate the coefficients for the selected frequency.
- Process the audio, save the filtered WAV, and compare spectra and listening results.

### Theory to explain
Digital filters, basic zeros and poles, FIR/IIR distinction, stability at an introductory level.

A notch filter has zero gain at one frequency `ω0 = 2π·f0/fs` and about unity gain elsewhere.

```
H(z) = (1 − 2cos(ω0) z⁻¹ + z⁻²) / (1 − 2r·cos(ω0) z⁻¹ + r² z⁻²)
```
- Zeros at `z = e^(±jω0)` on the unit circle make `H = 0` at ω0.
- Poles at `z = r·e^(±jω0)` just inside the circle nearly cancel the zeros everywhere except near ω0, so the gain stays near 1.
- The closer `r` is to 1, the narrower the notch. The notch bandwidth is about `BW = (1 − r)·fs/π` Hz.
- FIR has no feedback (only x terms). IIR has feedback (y terms). The notch is IIR.
- Stable if all poles are inside the unit circle: `r < 1`.

Note for your instructor's equations: the project says the filter structure/coefficient equations are provided. Use exactly those if they differ from the above (for example a different normalisation). The idea is the same: compute `b` and `a` from `f0`, `fs`, and a width parameter.

Choosing `r`: it depends on `fs`. With a low `fs` like 1000 Hz, `r = 0.95` is a reasonable notch. With `fs = 8000` or `44100`, `r = 0.95` gives a very wide notch. Choose a bandwidth in Hz and compute `r`, as below.

### Code

```python
f0 = 50                                      # Hz, the detected interference (from Part 3)
BW = 10                                      # desired notch bandwidth in Hz
r = 1 - np.pi * BW / fs                      # pole radius for that bandwidth
w0 = 2*np.pi*f0 / fs

# Coefficients (calculated, not copied from a library)
b = np.array([1, -2*np.cos(w0), 1.0])
a = np.array([1, -2*r*np.cos(w0), r**2])
b = b * (np.sum(a) / np.sum(b))              # scale so the gain at DC is exactly 1
print("b =", b); print("a =", a)

# Check against the SciPy design (optional): Q = f0 / BW
b_sp, a_sp = signal.iirnotch(f0, Q=f0/BW, fs=fs)
print("scipy b =", b_sp); print("scipy a =", a_sp)

# Stability: poles inside the unit circle
poles = np.roots(a)
print("pole magnitudes:", np.abs(poles), "-> stable:", np.all(np.abs(poles) < 1))

# --- Apply the filter ---
filtered = signal.lfilter(b, a, noisy)       # `noisy`, `fs` come from Part 3

# --- Save the output WAV ---
out = filtered / np.max(np.abs(filtered))
wavfile.write("filtered.wav", fs, (out * 32767).astype(np.int16))

# --- Spectra before and after ---
def amp_spectrum(sig):
    return np.fft.rfftfreq(len(sig), 1/fs), np.abs(np.fft.rfft(sig)) * 2 / len(sig)

f, S_noisy = amp_spectrum(noisy)
_, S_filt  = amp_spectrum(filtered)

wf, Hf = signal.freqz(b, a, worN=8192, fs=fs)

fig, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(f, S_noisy, label="noisy"); ax[0].plot(f, S_filt, label="filtered")
ax[0].set(xlim=(0, 300), title="Spectrum before and after the notch", xlabel="Hz", ylabel="amplitude")
ax[0].legend()
ax[1].plot(wf, 20*np.log10(np.abs(Hf) + 1e-12))
ax[1].set(xlim=(0, 300), ylim=(-60, 5), title="Notch filter frequency response", xlabel="Hz", ylabel="gain (dB)")
ax[2].plot(noisy[:int(0.05*fs)], label="noisy"); ax[2].plot(filtered[:int(0.05*fs)], label="filtered")
ax[2].set(title="Waveforms (first 50 ms)", xlabel="sample"); ax[2].legend()
plt.tight_layout(); plt.show()

# --- Numbers for the report ---
def peak_db(S, f, f0):
    m = (f > f0-3) & (f < f0+3)
    return 20*np.log10(np.max(S[m]) + 1e-12)
print(f"Tone level before: {peak_db(S_noisy, f, f0):.1f} dB | after: {peak_db(S_filt, f, f0):.1f} dB")

# Closeness to the original clean audio (x from Part 3)
def snr_db(ref, test): return 10*np.log10(np.sum(ref**2) / np.sum((ref - test)**2))
print(f"SNR noisy vs clean: {snr_db(x, noisy):.1f} dB | filtered vs clean: {snr_db(x, filtered):.1f} dB")

# --- Listening test (Jupyter/Colab) ---
from IPython.display import Audio, display
print("Noisy:");    display(Audio(noisy, rate=fs))
print("Filtered:"); display(Audio(filtered, rate=fs))
```

### Observations to write
- The 50 Hz spike is greatly reduced after filtering, and the rest of the spectrum is almost unchanged.
- In listening: the hum is gone or much quieter. The speech sounds almost the same.
- Limitations: the notch also removes a small part of the real signal near 50 Hz; there is a short start-up transient; a very narrow notch must match the tone frequency exactly, or the tone leaks through.
- Not required to reach any particular dB reduction. Report what you achieved.

---

## 8. Part 5: 2D Fourier and image filtering (Student 5)

### Tasks
- Load an image and convert it to grayscale.
- Compute and display the 2D Fourier spectrum using `fft2` and `fftshift`.
- Apply ideal circular low-pass and high-pass masks and display the filtered images.

### Theory to explain
2D Fourier transform, spatial frequency, low-pass and high-pass filtering.

```
F[u,v] = ΣΣ f[x,y] · e^(−j2π(ux/M + vy/N))
```
- Each point `(u, v)` is a spatial frequency: a wave pattern across the image with a direction and a density.
- After `fftshift`, the centre is zero frequency (DC, the average brightness).
- Near the centre: smooth areas. Far from the centre: edges and fine detail.
- We display `log(1 + |F|)` because DC is much larger than everything else, and without the log you would only see one dot.
- Ideal circular mask with cutoff `D0`: low-pass keeps `D <= D0` (blur), high-pass keeps `D > D0` (edges). `D` is the distance from the centre.
- Ideal masks have a sharp edge and cause ringing (ripples near edges) in the output.

### Code

```python
img = load_image()                           # grayscale float image, 256 x 256
M, N = img.shape

F = np.fft.fftshift(np.fft.fft2(img))        # centred 2D spectrum
spectrum = np.log1p(np.abs(F))               # log scale for display

# Distance of each frequency point from the centre (DC is at index M//2, N//2)
yy, xx = np.ogrid[:M, :N]
D = np.sqrt((yy - M//2)**2 + (xx - N//2)**2)

def lowpass(F, D0):
    return F * (D <= D0)

def highpass(F, D0):
    return F * (D > D0)

def to_image(Fm):
    return np.real(np.fft.ifft2(np.fft.ifftshift(Fm)))

# --- Original and spectrum ---
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img, cmap="gray", vmin=0, vmax=255); ax[0].set_title("Grayscale image"); ax[0].axis("off")
ax[1].imshow(spectrum, cmap="gray"); ax[1].set_title("2D spectrum log(1+|F|), centred"); ax[1].axis("off")
plt.tight_layout(); plt.show()

# --- Filters for several cutoffs ---
cutoffs = [10, 30, 60]
fig, ax = plt.subplots(2, len(cutoffs), figsize=(4*len(cutoffs), 8))
for j, D0 in enumerate(cutoffs):
    ax[0, j].imshow(to_image(lowpass(F, D0)), cmap="gray", vmin=0, vmax=255)
    ax[0, j].set_title(f"Low-pass, D0 = {D0}"); ax[0, j].axis("off")
    hp = to_image(highpass(F, D0))
    ax[1, j].imshow(np.abs(hp), cmap="gray")              # abs, because edges can be negative
    ax[1, j].set_title(f"High-pass, D0 = {D0}"); ax[1, j].axis("off")
plt.tight_layout(); plt.show()

# --- Show one mask too ---
plt.figure(figsize=(4, 4))
plt.imshow(D <= 30, cmap="gray"); plt.title("Low-pass mask, D0 = 30 (white = keep)"); plt.axis("off")
plt.show()
```

### Observations to write
- Low-pass: the image becomes blurrier as `D0` gets smaller, because fine detail is removed. Ringing may appear around edges because the mask is ideal.
- High-pass: only edges and details remain, and the large smooth areas become dark.
- Most of the energy is near the centre of the spectrum (low frequencies).
- Limitation: ideal masks produce ringing. A Gaussian mask would be smoother (optional).

---

## 9. Part 6: Image noise removal and evaluation (Student 6)

### Tasks
- Add periodic sinusoidal noise to a grayscale image.
- Display the noisy image spectrum and build a Notch Mask using the known added noise frequency.
- Apply the inverse 2D Fourier transform and compare original, noisy, and restored images.
- Calculate MSE and PSNR.

### Theory to explain
Periodic noise, notch filtering, inverse Fourier transform, MSE and PSNR.

```
noise(x,y) = A·sin(2π(u0·x/N + v0·y/M))
```
- A sine in the image is a stripe pattern. Its 2D spectrum consists of exactly two symmetric spikes at `(u, v) = ±(u0, v0)`.
- The image content lives mostly near the centre, so the two noise spikes stand out as bright dots.
- Because the Fourier transform is linear, setting those two small regions to 0 and applying `ifft2` removes the stripes, with a small loss of real image content nearby.
- Use whole-number `u0, v0` (complete cycles across the image), otherwise the energy spreads and you get smeared dots.
- You must notch both symmetric spots.

```
MSE  = (1/MN)·ΣΣ (original − restored)²
PSNR = 10·log10(255² / MSE)     [dB]
```
Smaller MSE means larger PSNR means better restoration. Compare against the original image (not the noisy one) and report PSNR for both the noisy and the restored image.

### Code

```python
img = load_image()                           # same image as Part 5
M, N = img.shape
yy, xx = np.mgrid[:M, :N]

# --- 1) Add periodic noise ---
A = 40                                       # noise amplitude
u0, v0 = 30, 20                              # whole cycles across width (u0) and height (v0)
noise = A * np.sin(2*np.pi*(u0*xx/N + v0*yy/M))
noisy = img + noise

# --- 2) Spectrum of the noisy image ---
F = np.fft.fftshift(np.fft.fft2(noisy))
spectrum = np.log1p(np.abs(F))

# --- 3) Notch mask at the two known spots ---
# After fftshift the DC is at (M//2, N//2); the noise spots are at (row, col) = (M//2 ± v0, N//2 ± u0)
R = 5                                        # notch radius in pixels
mask = np.ones((M, N))
spots = []
for s in (+1, -1):
    cy, cx = M//2 + s*v0, N//2 + s*u0
    spots.append((cy, cx))
    mask[(yy - cy)**2 + (xx - cx)**2 <= R**2] = 0

# --- 4) Inverse transform ---
restored = np.real(np.fft.ifft2(np.fft.ifftshift(F * mask)))

# --- 5) Evaluation ---
mse_n, psnr_n = psnr(img, noisy)
mse_r, psnr_r = psnr(img, restored)
print(f"Noisy:    MSE = {mse_n:.1f} | PSNR = {psnr_n:.2f} dB")
print(f"Restored: MSE = {mse_r:.1f} | PSNR = {psnr_r:.2f} dB")

# --- Plots ---
fig, ax = plt.subplots(2, 3, figsize=(14, 9))
ax[0, 0].imshow(img, cmap="gray", vmin=0, vmax=255);            ax[0, 0].set_title("Original")
ax[0, 1].imshow(np.clip(noisy, 0, 255), cmap="gray", vmin=0, vmax=255)
ax[0, 1].set_title(f"Noisy (PSNR {psnr_n:.1f} dB)")
ax[0, 2].imshow(np.clip(restored, 0, 255), cmap="gray", vmin=0, vmax=255)
ax[0, 2].set_title(f"Restored (PSNR {psnr_r:.1f} dB)")
ax[1, 0].imshow(spectrum, cmap="gray"); ax[1, 0].set_title("Noisy spectrum (two bright spots)")
ax[1, 1].imshow(mask, cmap="gray");     ax[1, 1].set_title("Notch mask (black = removed)")
ax[1, 2].imshow(spectrum, cmap="gray"); ax[1, 2].set_title("Spots marked")
for cy, cx in spots:
    ax[1, 2].add_patch(plt.Circle((cx, cy), R+2, fill=False, color="r"))
for a in ax.ravel():
    a.axis("off")
plt.tight_layout(); plt.show()
```

### Observations to write
- The noisy spectrum shows two bright symmetric dots at `±(u0, v0)` away from the centre. These are the stripes.
- After removing them, the stripes are gone and PSNR increases (report the two numbers).
- Restoration is not perfect: the notch also removes a little real image content, and clipping the noisy image to 0..255 adds small errors. That's expected, and the project does not require perfect denoising.
- Experiment worth showing: notch radius `R` too small leaves stripes; `R` too large blurs real detail.

---

## 10. Final summary cell (master notebook)

Add this at the end so the instructor sees the key results in one place (run after all parts):

```python
print("=== Project summary ===")
print(f"Part 3: detected interference at {f_peak:.1f} Hz (fs = {fs} Hz)")
print(f"Part 4: tone level before {peak_db(S_noisy, f, f0):.1f} dB, after {peak_db(S_filt, f, f0):.1f} dB")
print(f"Part 6: PSNR noisy {psnr_n:.2f} dB -> restored {psnr_r:.2f} dB")
```

Beware: if two parts use the same variable names (`x`, `noisy`, `f`, `F`, `img`), a later part overwrites the earlier one. Part 6 redefines `noisy` and `F`, and Part 4's summary values are computed earlier, so this cell still works as written. If someone changes the order, rename variables with suffixes (`noisy_audio`, `noisy_img`).

---

## 11. Discussion preparation

Format from the project file: each student explains their concept, shows the code/output, and answers one short question. The team also runs the complete project once.

Suggested presentation order: Part 1 to Part 6, about 2 minutes each.

For your own part, be able to answer:

| Part | Likely questions |
| --- | --- |
| 1 | What does ω mean? Why do peaks appear at ±ω0? Why is phase messy? What happens if N is smaller? |
| 2 | Why zero-pad to `N+M−1`? What does the moving average do in frequency? When is `y[n]=a y[n−1]+x[n]` stable? What is an impulse response? |
| 3 | What is Nyquist? How do you find the interference frequency? Why is the hum invisible in the waveform but visible in the spectrum? |
| 4 | Where are the zeros and poles? What does `r` control? Why IIR? Why must `r < 1`? What if the notch frequency is slightly wrong? |
| 5 | What is spatial frequency? Why `log(1+|F|)`? What do low-pass and high-pass do to an image? What is ringing? |
| 6 | Why two spots? Why integer `u0, v0`? What do MSE and PSNR measure? Why is PSNR not infinite after restoration? |

Small live changes the instructor might ask for (practise them): change the interference to 60 Hz and re-run Part 3 and 4; change the notch radius in Part 6; change the moving average length from 5 to 9; change the low-pass cutoff `D0`; change `r` or `BW`.

Everyone should also be able to say in one sentence what the other five parts do.

---

## 12. Final checklist

- [ ] Real `audio.wav` and `image.png` from the instructor are used (not the synthetic fallbacks)
- [ ] Kernel > Restart and Run All completes with no errors
- [ ] Every plot has a title and axis labels
- [ ] Each part has a short theory cell and an observations cell
- [ ] `filtered.wav` exists and sounds better than `noisy.wav`
- [ ] PSNR (noisy and restored) and the tone levels (before and after) are written down
- [ ] Cover sheet filled in: university, faculty, names, IDs
- [ ] Notebook saved with outputs, plus a PDF export as a backup
- [ ] Every member can explain their own part and the overall flow

## 13. Common problems

| Problem | Fix |
| --- | --- |
| `wavfile.read` returns two columns | The audio is stereo: average the channels (already handled in `load_audio`). |
| Audio sounds distorted after saving | Scale to int16 as shown: `(out/np.max(np.abs(out))*32767).astype(np.int16)`. |
| Notch removes a big range around 50 Hz | `r` is too small for your `fs`. Use `r = 1 − π·BW/fs`. |
| Notch does not remove the tone | The notch frequency doesn't match the tone: use the frequency measured in Part 3. |
| Image spectrum shows only one white dot | Display `np.log1p(np.abs(F))` instead of `np.abs(F)`. |
| Noise spots don't line up with the mask | Use whole-number `u0, v0`, an even image size, and `fftshift` before building the mask. |
| Restored image looks very dark or bright | Clip to 0..255 and show with `vmin=0, vmax=255`. |
| DTFT cell is very slow | Reduce `N` or `K`. The double loop is expected to be slow. |
| A part works alone but fails in the merged notebook | A variable was overwritten or is missing: define it in that part's first cell. |
