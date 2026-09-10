# Search-Based Adaptive Image Compression and Resizing

[![tests](https://github.com/XSHRADER/PixelOpt/actions/workflows/tests.yml/badge.svg)](https://github.com/XSHRADER/PixelOpt/actions/workflows/tests.yml)

Fit an image into a byte budget at the highest measured quality. The compressor
chooses an output resolution, an encoder (JPEG or WebP) and a quality setting,
then verifies its choice with SSIM rather than assuming it.

## How it decides

**Resolution comes from arithmetic, not a guess.** After downscaling by `r` the
image has `r² · N` pixels, so an encoder given a fixed byte budget works at
`bpp / r²`. Solving for the rate we actually want gives `r = sqrt(bpp / 0.55)`,
capped at 1.0. Below roughly 0.35 bpp a block-transform codec spends its whole
budget on artefacts and downscaling first preserves more real detail than the
lost resolution costs; above roughly 0.5 bpp full resolution always wins.

**Quality is solved, not searched.** Encoded size is monotonic in the quality
setting and roughly exponential in it, so interpolating between two bracketing
probes on log-size converges in about three encodes per candidate.

**The winner is measured.** The seed resolution, its two neighbours and full
resolution are each encoded in both formats, and SSIM picks the survivor.
"Largest file under the budget" is not the same thing as "best-looking file
under the budget" — on a high-detail image those two answers differ by 9% SSIM
at the same byte count.

Full resolution is always tried even when the bpp rule says to shrink: the rule
assumes the image needs 0.55 bpp, but an easily compressed one needs far less
and often fits the budget untouched.

## Measured results

`python benchmark.py` runs four 1200×1200 test images across five target sizes.

| image | target | actual | error | SSIM | PSNR | vs PNG | encodes |
|---|---|---|---|---|---|---|---|
| gradient | 25 KB | 18.7 KB | −25.0% | 0.995 | 52.0 | 1.1x | 31 |
| gradient | 50 KB | 48.3 KB | −3.4% | 0.998 | 57.3 | 0.4x | 24 |
| edges/text | 25 KB | 23.5 KB | −6.1% | 0.999 | 48.7 | 0.3x | 31 |
| high detail | 25 KB | 24.5 KB | −1.9% | 0.446 | 21.4 | 159.0x | 21 |
| high detail | 50 KB | 48.6 KB | −2.9% | 0.476 | 21.6 | 80.2x | 30 |
| high detail | 100 KB | 98.9 KB | −1.1% | 0.585 | 22.0 | 39.4x | 19 |
| high detail | 200 KB | 198.1 KB | −1.0% | 0.705 | 23.1 | 19.7x | 23 |
| high detail | 500 KB | 495.0 KB | −1.0% | 0.967 | 32.0 | 7.9x | 21 |
| near-flat | 25 KB | 23.0 KB | −8.0% | 0.990 | 50.3 | 34.8x | 22 |

**Size-targeting error: median 2.4%, mean 5.5%. All 20 runs landed at or under
target** — the point of a size budget is not to exceed it. Median SSIM 0.993,
median 20 encoder calls per run.

The "vs PNG" column compares against a lossless PNG of the same image, not
against raw RGB bytes. That distinction matters: measuring against raw RGB is
what let an earlier version of this table advertise 46x. Only 11 of the 20 runs
beat PNG at all — on synthetic gradients and glyph-like art, a lossless encoder
is simply the better tool, and the benchmark now says so instead of hiding it
behind an inflated baseline.

Twelve of the twenty runs are omitted above because their target was
**unreachable**: at full resolution and maximum quality the encoder simply
cannot produce more bytes. A near-flat image tops out at 23.9 KB, so asking for
500 KB cannot produce 500 KB. Those runs return the maximum-fidelity encoding
available and the benchmark reports them separately rather than counting them
as error. Undershooting there is a property of the image, not a miss.

The test images are generated rather than photographed, and deliberately cover
cases that stress a compressor differently. Size-targeting error transfers to
real photographs; the absolute SSIM/PSNR figures are only meaningful relative
to one another. The low SSIM on the high-detail image is expected — dense
high-frequency noise is what a lossy codec discards first.

### What changed, and what it bought

The previous version predicted resize and quality with a CatBoost regressor,
then swept the entire resize grid and kept whichever encoding used the most
bytes under the budget.

Two problems. The model could not change the output: the search explored every
grid point regardless of the prediction and ranked the results by a criterion
that never referenced it, so three different predictions produced byte-identical
files. And "most bytes under budget" is not a quality criterion.

Replacing it with the bpp rule, log-size interpolation and SSIM ranking, on the
same test images and targets:

| case | before | after | change |
|---|---|---|---|
| high detail @ 100 KB | 0.524 | 0.585 | +11.6% SSIM |
| high detail @ 200 KB | 0.612 | 0.705 | +15.2% SSIM |
| high detail @ 500 KB | 0.902 | 0.967 | +7.2% SSIM |
| edges/text @ 100 KB | 0.988 @ 99.6 KB | 0.999 @ 23.5 KB | better, 4× smaller |
| near-flat @ 100 KB | 0.990 @ 91.7 KB | 0.990 @ 23.9 KB | same, 3.8× smaller |
| encoder calls per run | 69 | 17.6 | 3.9× fewer |

The dependency footprint went with it: CatBoost, scikit-learn, joblib, pandas
and matplotlib are all gone, along with a 3 MB committed model file.

## Features

- Picks output resolution from the bits-per-pixel budget in closed form.
- Solves encoder quality by interpolation on log file size.
- Encodes JPEG (progressive and optimized) and WebP, keeping whichever scores
  higher at the same budget.
- Ranks candidates by measured SSIM instead of file size.
- Honours EXIF orientation, so portrait photos are not returned sideways.
- Composites transparency onto white instead of silently dropping the alpha.
- Reports SSIM, PSNR and MSE against the original at full resolution.
- Hands the download button the exact bytes that were measured.
- Includes a Streamlit app, a batch CLI, and a reproducible benchmark.

## Project layout

- `pixelopt/adaptive_compressor.py` — the search and the encoding pipeline.
- `pixelopt/image_features.py` — loading, content statistics, quality metrics.
- `pixelopt/cli.py` — command-line entry point.
- `app.py` — Streamlit web interface.
- `benchmark.py` — reproducible measurements behind the table above.
- `tests/test_compression.py` — unit tests.

[ARCHITECTURE.md](ARCHITECTURE.md) covers how the search works and why, and
what to know before changing it.

## Command line

```bash
pip install -e .

pixelopt photo.jpg --target 200
pixelopt *.jpg --target 150 --out-dir web/ --format webp
```

Without installing, `python -m pixelopt` works the same way from a clone.

```text
target 60 KB

texture.png       1970.3K ->    57.0K  WEBP     900x900   SSIM 0.582   34.5x smaller
glyphs.png           7.6K ->    37.7K  WEBP     900x900   SSIM 0.994   LARGER than input
blobs.png         1240.9K ->    59.9K  JPEG     900x900   SSIM 0.976   20.7x smaller
```

Output never exceeds the target, but re-encoding a file that already fits can
make it bigger — a lossless PNG of flat or glyph-like art beats any lossy
encoder — so the CLI says so rather than reporting "0.2x smaller". Existing
files are skipped unless you pass `--overwrite`, and a failure on one input does
not abort the batch.

## Quick start

1. Open a terminal in the project folder.

2. Use the built-in launcher on Windows:

   ```bat
   launch_app.bat
   ```

   Or run manually:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8501
   ```

3. Open the browser at `http://127.0.0.1:8501`.

4. Upload an image, choose the target output size in KB, and pick a format
   (or leave it on Auto).

5. Click **Compress and resize image**, then download either output.

There is no training step and no model file — the app runs straight from a
clone.

## Execution flow

```text
User uploads image
       ↓
app.py  (EXIF-corrected load, transparency warning)
       ↓
AdaptiveImageCompressor.compress()
       ↓
seed_resize()  — resolution from the bits-per-pixel budget
       ↓
_best_quality()  — quality by interpolation on log-size, per candidate
       ↓
SSIM ranking across resolutions and formats
       ↓
SSIM / PSNR / MSE measured on the winner at full resolution
       ↓
Resized and compressed outputs shown and downloadable
```

## Typical usage

- Upload a photo or product image.
- Choose a target size such as 200 KB.
- The compressor selects resolution, format and quality, and reports what the
  chosen settings actually cost in SSIM, PSNR and MSE.
- Download the resized file or the final compressed version.

## Future extensions

- AVIF and JPEG XL support.
- Chroma-aware quality metrics — SSIM is currently computed on luma only.
- Per-image tuning of the 0.55 bpp constant, which is a corpus-wide average.
- Fall back to lossless PNG when it beats the lossy encoders at the budget.
