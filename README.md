# Lossless Audio Compression

A complete **lossless audio compression system** built from scratch in Python with **Rice coding** and **fixed linear prediction**. It compresses WAV files into a custom `.ex2` format, decodes them back, and checks that the decoded file is **byte-for-byte identical** to the original.

Exercise 2 of the Intelligent Signal Processing final coursework.

## How it works

```
WAV samples ──► fixed predictor ──► residuals ──► ZigZag ──► Rice code ──► .ex2 file
                (order 0–4)        (actual − predicted)  (signed → unsigned)  (unary q + K-bit r)

.ex2 file ──► Rice decode ──► ZigZag⁻¹ ──► inverse predictor ──► WAV (identical to the original)
```

1. **Bit-level I/O**: custom `BitWriter` and `BitReader` classes write and read single bits. The unary encoder and decoder have fast paths: whole `0xFF` bytes, and a lookup table that counts leading 1s.
2. **Fixed predictors (orders 0–4)**: each sample is predicted from the previous ones, for example order 2 is `2·x[n−1] − x[n−2]`. Only the **residual** (actual − predicted) is stored. Mono and multi-channel audio are both supported.
3. **Choosing the best predictor**: for each K, every order is tried and the exact Rice bit cost is computed, `Σ(q + 1 + K)`. The cheapest order is used.
4. **ZigZag encoding**: maps signed residuals to unsigned integers (0, −1, 1, −2, … → 0, 1, 2, 3, …) so they can be Rice-coded.
5. **Rice coding**: each value is split into a quotient `q = x >> K`, written in unary, and a K-bit remainder.
6. **`.ex2` file format**: a 28-byte header holds the magic `EX2R`, the version, K, the predictor order, channels, sample width, sample rate, frame count, sample count and bit count. The packed bitstream follows.
7. **Verification**: the decoded WAV is compared byte-for-byte with the original.

8-, 16-, 24- and 32-bit PCM WAV files are supported.

## Results

| File | Original size (bytes) | Rice (K = 4) | Rice (K = 2) | % Compression (K = 4) | % Compression (K = 2) | Lossless |
|---|---|---|---|---|---|---|
| Sound1.wav | 1,002,088 | 706,520 | 1,828,615 | **+29.50%** | −82.48% | ✓ |
| Sound2.wav | 1,008,044 | 39,102,997 | 155,435,737 | −3779.10% | −15319.54% | ✓ |

Best predictor order: **2** for Sound1 and **0** for Sound2, for both K values.

**Why Sound2 gets bigger:**
- **Sound1** is quiet and smooth: neighbouring samples differ by about 70 on average, so prediction leaves small residuals and K = 4 compresses it by about 30%.
- **Sound2** is loud, noisy and uses the full 16-bit range: neighbouring samples differ by about 7,000. No predictor helps, so the residuals stay large.
- **The cost of a small K:** with K = 2 or 4, each large residual needs hundreds of unary bits for its quotient.

The system is still **lossless in every case**. The results show how strongly Rice coding depends on choosing K to suit the signal; a K of around 12–13 would suit Sound2.

The script also plots **histograms of the residuals (deltas)** for each file and K value.

## Running it

### Quick start (one command)

Run this command in the terminal. It downloads the project from GitHub into a temporary folder, installs the required packages in a separate environment (so your main Python isn't changed), and runs the full encode → decode → verify pipeline:

```bash
D=$(mktemp -d) && gh repo clone Alizea2/ISP-Lossless-Audio-Compression "$D" && cd "$D" && python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt && .venv/bin/python "Exercise 2.py"
```

It takes about 40 seconds:
- **Terminal:** shows the chosen predictors, the compressed sizes, a lossless check for each file, and the results table.
- **Output files:** written to `exercise2_output/`. These are the `.ex2` files and the decoded `_Enc_Dec.wav` files.
- **Histograms:** four windows open one after another. Close each one to see the next.

> This needs Python 3 and the [GitHub CLI](https://cli.github.com/) (`gh`) signed in to an account that can access this repository. The outputs take about 200 MB of disk space, mostly the Sound2 K = 2 file.

### Manual setup

From inside the project folder:

```bash
pip install -r requirements.txt
python "Exercise 2.py"
```

`Exercise 2.py` was exported from a Jupyter notebook. The `# In[n]:` markers show the original cells, so the code can also be pasted into a notebook cell by cell.

## Project Structure

| File / Folder | Purpose |
|---|---|
| `Exercise 2.py` | The whole system: bit I/O, ZigZag, Rice coding, predictors, the `.ex2` format, the pipeline and the plots |
| `Exercise2_Files/` | Input audio: `Sound1.wav`, `Sound2.wav` |
| `exercise2_output/` | Created when the script runs; not stored in the repo because the files are large |
| `requirements.txt` | Python packages |

## Built With

- Python 3, [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/)
- The standard library `wave` and `struct` modules for WAV and binary I/O

## Related exercises

- [ISP-Audio-Effects-App](https://github.com/Alizea2/ISP-Audio-Effects-App)
- [ISP-Audio-Captcha-Voice-Control](https://github.com/Alizea2/ISP-Audio-Captcha-Voice-Control)
- [ISP-Audio-Steganography](https://github.com/Alizea2/ISP-Audio-Steganography)
- [ISP-Airport-Speech-Recognition](https://github.com/Alizea2/ISP-Airport-Speech-Recognition)

## Author

[@Alizea2](https://github.com/Alizea2)
