# Multi-Domain DSP Project

A six-part digital signal processing project covering mathematical signals, recorded audio, and digital images.

## Team

- Ahmed Mohamed Ramadan — 212300037
- Omar Ashraf Abdelhady — 212231416
- Abdullah Hussein Sayed — 212300261
- Youssef Nasser Sayed Ahmed — 212300554

## Project contents

- `dsp_project.ipynb` — complete notebook with explanations, code, plots, and results
- `audio.wav` — original voice recording used by the audio sections
- `image.png` — source image used by the image-processing sections
- `noisy.wav` — recording after adding simulated 50 Hz interference
- `filtered.wav` — result after applying the notch filter
- `DSP_Project_Guide.md` — detailed project requirements and reference guide

## Run locally

1. Install Python 3.11 or newer.
2. Open a terminal in this repository.
3. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start Jupyter:

   ```bash
   jupyter notebook
   ```

6. Open `dsp_project.ipynb` and select **Run All**.

The notebook must be run from the repository directory so it can find `audio.wav` and `image.png`.

## Main results

- The manual DTFT agrees with NumPy's FFT at the FFT sample frequencies.
- Zero-padded FFT multiplication matches linear convolution.
- The audio notch filter reduces the added 50 Hz tone by about 33 dB.
- Image restoration improves PSNR from about 19.17 dB to 41.64 dB.

