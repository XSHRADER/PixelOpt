# PixelOpt — search-based image enhancement and compression

[![tests](https://github.com/XSHRADER/PixelOpt/actions/workflows/tests.yml/badge.svg)](https://github.com/XSHRADER/PixelOpt/actions/workflows/tests.yml)

Fit an image into a byte budget at the highest measured quality. PixelOpt
chooses an output resolution, an encoder (JPEG, WebP or lossless PNG) and a
quality setting, optionally cleans the image up first, and verifies every
choice with SSIM rather than assuming it.

## Enhancement, and why it belongs before the encoder

**Denoising pays for itself.** Noise is incompressible high-frequency data —
the most expensive thing an encoder can be asked to store. Measured against a
clean ground truth that neither pipeline ever saw, denoising before compression
improved SSIM in **9 of 9 tested conditions**:

| noise (sigma) | 40 KB budget | 80 KB | 160 KB |
|---|---|---|---|
| 8 | +6.2% | +20.5% | +32.2% |
| 16 | +30.9% | +80.1% | +132.3% |
| 28 | +110.8% | +282.4% | +383.5% |

The effect gets *stronger* at larger budgets, which is backwards until you see
why: a generous budget lets the encoder faithfully reproduce the noise, while a
tight one discards it as a side effect. Compression was accidentally denoising
at low bitrates, and doing it deliberately is strictly better. End to end on a
noisy photo at 80 KB, fidelity went **0.692 → 0.983**.

**Auto mode measures rather than guesses.** Immerkaer noise estimation picks
the filter strength, and does nothing at all below a threshold. It deliberately
never touches contrast, white balance or saturation — those change how a
photograph is meant to look, which is the photographer's call, not the tool's.

Sharpening is the counterpart: it *adds* high-frequency detail the encoder then
has to store. Worth it after a downscale, which softens edges, but not free.

### The metric had to split first

The old design measured quality as SSIM against the image the compressor was
handed. Enhancement deliberately changes that image, so sharing one reference
would have made the metric punish the enhancement and the search fight it.

```
original ──enhance──> reference ──compress──> output
             │                        │
             │                        └── fidelity: SSIM(reference, output)
             │                            what the ENCODER lost
             └── drift: SSIM(original, reference)
                 what the ENHANCEMENT changed
```

Drift is not an error — a large drift on a noisy photo is the denoiser doing
its job — but it is also what overprocessing looks like, so both are reported.

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

- Denoises, sharpens, white-balances, levels and adjusts contrast before
  encoding, with an auto mode driven by a noise measurement.
- Reports fidelity and drift separately, so enhancement and compression are
  never conflated.
- Picks output resolution from the bits-per-pixel budget in closed form.
- Solves encoder quality by interpolation on log file size.
- Encodes JPEG (progressive and optimized), WebP and lossless PNG, keeping
  whichever scores highest at the same budget. On glyph-like art PNG returns
  2.8 KB at SSIM 1.0 where the lossy path produced 23.5 KB.
- Plots quality against budget and marks the knee, so a size can be chosen
  from evidence instead of guessed.
- Ranks candidates by measured SSIM instead of file size.
- Honours EXIF orientation, so portrait photos are not returned sideways.
- Composites transparency onto white instead of silently dropping the alpha.
- Reports SSIM, PSNR and MSE against the original at full resolution.
- Hands the download button the exact bytes that were measured.
- Form photo mode: exact dimensions, a size range that holds under both
  kilobyte readings, and a crop that follows a face or the detail.
- Batch mode in the app, with failures isolated and a ZIP plus CSV report.
- A full-resolution damage heatmap and damaged-area measurement.
- A Streamlit app with top navigation, a batch CLI, and a reproducible
  benchmark.

## Form photo mode

Upload forms ask for things a general compressor never handles: *a JPEG of
exactly 200 × 230 pixels, between 20 and 50 KB*. Form photo mode meets that.

- **Exact dimensions.** The crop anchors on a detected face, or on the most
  detailed region, rather than blindly taking the centre. "Fit on white" keeps
  everything, which is safer for signatures.
- **A size range, not just a ceiling.** Portals disagree on whether a KB is
  1000 or 1024 bytes, which is a common reason for a rejected upload. The limits
  are applied so the file passes under *both* readings: the maximum in 1000-byte
  KB, the minimum in 1024-byte KB.
- **Honest padding.** An image too simple to reach the minimum even at quality
  100 is padded with JPEG comment bytes. The pixels are untouched, and the app
  reports exactly how many bytes were added.
- **Portal-safe output.** Baseline (not progressive) JPEG, optional grayscale,
  and the DPI written into the file.

Presets cover an application photo, signature, passport 35 × 45 mm, a 2 × 2 in
square and a profile thumbnail. They are common sizes, not any organisation's
official specification.

## Batch mode

Upload many images, apply one set of settings — a target size or a form
preset — and download a ZIP plus `report.csv`. A corrupt file is recorded and
the rest carry on. `photo.png` and `photo.jpg` get distinct output names, not
one silently overwriting the other. The ZIP holds exactly the bytes the report
describes.

## Damage heatmap

A single SSIM number cannot say *where* quality was lost: the same 0.97 can be
damage spread thinly everywhere or a ruined face on an otherwise clean frame.
SSIM is the mean of a local similarity map, so PixelOpt keeps the map. The
viewer overlays it (brighter = more loss), outlines the worst region, and
reports the **damaged area**: the share of the frame whose local SSIM fell below
0.9, roughly where damage becomes visible at 1:1.

The map is measured at full resolution. Downscaling first averages the
artefacts away before SSIM can see them: on a faded scan the damaged share read
86% at full resolution, 52% after a barely visible downscale, and 0.4% at
400k pixels, all for the same file.

## Project layout

- `pixelopt/adaptive_compressor.py` — the search and the encoding pipeline.
- `pixelopt/enhance.py` — enhancement operations and noise measurement.
- `pixelopt/pipeline.py` — enhance-then-compress, and the two-reference split.
- `pixelopt/forms.py` — form photo mode: framing, both KB readings, padding.
- `pixelopt/analysis.py` — the damage map, its summary and the heatmap.
- `pixelopt/batch.py` — running a job over many files, ZIP and CSV export.
- `pixelopt/image_features.py` — loading, content statistics, quality metrics.
- `pixelopt/cli.py` — command-line entry point.
- `app.py` — Streamlit entry point: theme, styling, top navigation.
- `app_pages/` — the Optimize, Form photo and Batch pages.
- `app_shared.py` — cached Streamlit helpers shared by the pages.
- `ui_components.py` — hero, count-up metrics, range meter and the
  slide / flicker / heatmap comparison viewer (Components v2).
- `benchmark.py` — reproducible measurements behind the table above.
- `tests/test_compression.py` — unit tests.

[ARCHITECTURE.md](ARCHITECTURE.md) covers how the search works and why, and
what to know before changing it.

## Command line

```bash
pip install -e .

pixelopt photo.jpg --target 200
pixelopt *.jpg --target 150 --out-dir web/ --format webp
pixelopt scan.png --target 300 --enhance scan
pixelopt noisy.jpg --target 80 --enhance auto
```

Enhancement presets are `none`, `photo`, `scan`, `screenshot`, `vivid`, plus
`auto`. Individual flags (`--denoise`, `--sharpen`, `--contrast`,
`--saturation`, `--white-balance`, `--auto-level`) layer on top of a preset.

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

2. Launch it — one command does everything:

   ```bash
   python launch.py
   ```

   On Windows you can double-click `launch_app.bat`; on macOS or Linux run
   `./launch.sh`. The launcher checks Python, creates or reuses `.venv`,
   installs dependencies only when `requirements.txt` changes, runs an engine
   self-test, picks a free port, waits until the app answers its health check,
   opens your browser, and restarts PixelOpt if it crashes. Running it again
   while PixelOpt is up just reopens the existing instance.

   | option | what it does |
   |---|---|
   | `--check` | run every test suite before launching |
   | `--once --no-browser` | launch, confirm it is healthy, shut down — a smoke test |
   | `--port 8600` | preferred port; the next free one is used if it is busy |
   | `--reinstall` | reinstall dependencies even if unchanged |
   | `--install-autostart` | start PixelOpt when you log in (Windows, Linux) |
   | `--remove-autostart` | undo that |

   Logs go to `logs/`. If anything fails, the launcher stops at that stage and
   prints the reason and the last lines of the log.

   Or run manually:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8501
   ```

3. Open the browser at `http://127.0.0.1:8501`. The top navigation switches
   between **Optimize**, **Form photo** and **Batch**.

4. Upload an image, choose the target output size in KB, and pick a format
   and enhancement preset (or leave both on Auto).

5. Compare the result against the reference with the divider, **zooming to 1:1
   or past it** — at fit-to-screen a 200 KB and a 500 KB encode of the same
   photo look identical, so a comparison that cannot zoom proves nothing.

6. Optionally measure the budget curve to see where the knee is, then
   download.

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
- Chroma-aware quality metrics — SSIM is currently computed on luma only, so
  chroma artefacts are invisible to the ranking.
- Per-image tuning of the 0.55 bpp constant, which is a corpus-wide average.
- Responsive image sets with a generated `srcset` snippet.
- Face-aware budget allocation in the general search (form mode already
  crops around faces).
- A faster denoiser for very large images: non-local means costs ~0.85s at
  0.8 MP and scales with pixel count.
