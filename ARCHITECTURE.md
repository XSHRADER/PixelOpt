# Architecture

How the compressor is put together, and why it is built this way. For usage,
see the [README](README.md).

## The problem

Given an image and a byte budget, choose an output resolution, an encoder and a
quality setting that maximise perceived quality without exceeding the budget.

Three variables, one hard constraint, and an objective (perceived quality) that
can only be measured after encoding. The design question is how much of that
search to do by measurement and how much by arithmetic.

## The three decisions

### Resolution: arithmetic

After downscaling by `r`, the image has `r² · N` pixels, so an encoder handed a
fixed byte budget works at `bpp / r²` bits per pixel. Solving for the rate we
want gives:

```
r = min(1.0, sqrt(bpp / BPP_TARGET))        BPP_TARGET = 0.55
```

`BPP_TARGET` is where a block-transform codec stops producing artefacts and
starts producing detail. Below roughly 0.35 bpp, downscaling first preserves
more real information than the lost resolution costs; above roughly 0.5 bpp,
full resolution always wins. The constant is a corpus-wide average, not a
per-image optimum — see *Known limitations*.

`seed_resize()` in `pixelopt/adaptive_compressor.py`.

### Quality: solved, not searched

Encoded size is monotonic in the quality setting and roughly exponential in it.
So instead of sweeping or bisecting, probe the quality floor and ceiling and
interpolate on **log** size:

```
fraction = (log(target) - log(size_low)) / (log(size_high) - log(size_low))
```

This converges in about three probes per candidate where bisection needs seven.
`_best_quality()` returns `None` when the image will not fit even at the floor,
which is how the caller learns a resolution is infeasible.

### The winner: measured

The bpp rule gives a seed. The search then encodes the seed, its two grid
neighbours **and full resolution** in every enabled format, and ranks the
survivors by SSIM.

Full resolution is always included because the rule assumes the image needs
0.55 bpp — an easily compressed one needs far less and often fits the budget
untouched. Testing it anyway lifted a glyph-like test image from 0.956 to 0.999
SSIM *and* produced a smaller file, for two extra encodes.

Ranking is by measured quality rather than by file size because "largest file
under the budget" is not the same thing as "best-looking file under the budget".
On a high-detail image at a 150 KB budget, those two criteria differ by 9% SSIM.

## Two subtleties worth knowing before you change the ranking

**The proxy flatters downscaling.** Full-resolution SSIM costs ~3.8 s on a 12 MP
image, so candidates are scored against a size-capped grayscale proxy of the
original. Because that proxy is itself downscaled, a downscaled candidate is
being compared against a similarly smoothed reference and scores slightly too
well. On a near-flat image a 0.9 encode "beat" full resolution by 0.0004 SSIM
for exactly this reason. `SCORE_TOLERANCE` exists to absorb it: inside the
tolerance band, prefer resolution, then fewer bytes.

**The winner's reported metrics are not the ranking score.** Ranking uses the
proxy; the returned `ssim`/`psnr`/`mse` are measured once, at full resolution,
on the winner. Do not conflate them.

## The enhancement stage, and the reference split

Enhancement runs before the encoder, and it forced a change to how quality is
measured. The compressor scores SSIM against the image it was handed;
enhancement deliberately changes that image. Sharing one reference would make
the metric punish the enhancement and the search fight it, so `pipeline.py`
splits it: **fidelity** is `SSIM(reference, output)` and **drift** is
`SSIM(original, reference)`.

This is the single most important thing to understand before extending the
tool. Any new operation that alters pixels belongs before the compressor and
is accounted for in drift, not fidelity.

Two operations change how many bits the image needs, which is why they are
here rather than in a post-processing step:

- **Denoising reduces them.** Noise is incompressible. Measured against a
  clean ground truth neither pipeline saw, denoise-then-compress won 9 of 9
  conditions, from +6% to +383% SSIM. Auto mode sizes the filter from an
  Immerkaer noise estimate, which reads about 0.66x true sigma -- the 0.83
  factor in `auto_enhancements` divides that out.
- **Sharpening increases them.** An unsharp mask adds high-frequency detail
  the encoder must then store.

Auto mode never touches contrast, white balance or saturation. Those change
how a photograph is meant to look, and that is not the tool's call.

## Module map

| file | responsibility |
|---|---|
| `pixelopt/adaptive_compressor.py` | the search, the encoders, the public `compress()` |
| `pixelopt/enhance.py` | enhancement operations, noise estimation, presets |
| `pixelopt/pipeline.py` | enhance-then-compress, the reference split, the R-D curve |
| `pixelopt/image_features.py` | loading (EXIF, transparency), content stats, quality metrics |
| `pixelopt/cli.py` | command-line entry point, batch handling |
| `app.py` | Streamlit interface |
| `ui_components.py` | the zoomable before/after comparison (Components v2) |
| `benchmark.py` | the measurements the README publishes |
| `tests/test_compression.py` | compressor and loader tests |
| `tests/test_enhance.py` | enhancement, pipeline and curve tests |

## Call order

```text
compress(image, target_kb, image_format)
  ├── load_image()                  EXIF orientation, alpha composited onto white
  ├── seed_resize()                 resolution from the bits-per-pixel budget
  ├── _candidates()                 seed, its two neighbours, and 1.0
  └── for each format, for each candidate resolution:
        ├── _best_quality()         log-size interpolation → highest fitting quality
        └── _score()                SSIM against the capped proxy
      → best score wins, ties inside SCORE_TOLERANCE prefer resolution then fewer bytes
  └── compute_quality_metrics()     SSIM / PSNR / MSE on the winner, full resolution
```

## Design decisions and their reasons

**Why no machine-learning model.** There was one: a CatBoost regressor
predicting resize factor and quality. It was removed because it could not affect
the output — the old search explored the whole grid regardless of the prediction
and ranked by a criterion that never referenced it, so three different
predictions produced byte-identical files. Measured against the bpp rule, a
perfect resize predictor had roughly 1.5% SSIM of headroom to offer, which does
not pay for a training corpus, a labelling run and a 3 MB artefact. Feature
extraction alone cost 1.75 s on a 12 MP image — ten times the entire search it
would have replaced.

If you want to revisit this, the honest experiment is to measure the oracle-vs-rule
gap on a few hundred **real photographs** first. Synthetic images will teach a
model the generator.

**Why SSIM and not a better perceptual metric.** SSIM is in `scikit-image`,
which is already a dependency. It is computed on luma only, so chroma artefacts
are invisible to the ranking. This is a real weakness, listed below.

**Why always scale from the original.** An earlier version resized an
already-resized image, so the model's factor and the search's factor multiplied.
The search could only shrink and never climb back toward a larger target; a
gradient asked for 100 KB, 200 KB and 500 KB returned the same 28 KB file.
`_encode()` takes the original every time.

**Why the download button gets `raw_bytes`.** It used to re-encode the image at
the reported quality, producing a different file from the one that was measured.
The exact measured byte string is now what leaves the process.

## Notes on the Streamlit app

Three things there are load-bearing and easy to undo by accident:

- **The comparison component must be able to zoom.** At fit-to-screen a 200 KB
  and a 500 KB encode of the same photo are indistinguishable; the artefacts
  are at the pixel level. The component keeps one shared transform across both
  image layers, with the divider as a clip on an *untransformed* wrapper, so
  the two views cannot drift out of alignment. Its "after" layer is the real
  encoded bytes, so what is inspected at 4:1 is what gets downloaded.
- **The budget curve is gated behind a button.** Streamlit computes the
  contents of hidden tabs, so without the guard eight extra compressions ran
  on every rerun even while another tab was on screen. `st.stop()` is not an
  option inside a tab -- it halts the whole script and takes the later tabs
  with it.
- **The curve's budget ladder comes from the achievable ceiling**, not from
  the slider. Spacing it off the slider put most of the ladder above what a
  well-compressing image can produce, leaving nothing to plot. The ceiling
  probe excludes lossless, because a PNG ceiling is not representative of a
  lossy rate-distortion curve.

Theming is in `.streamlit/config.toml` rather than injected CSS, so it applies
to every element and survives upgrades. The only hand-written CSS lives inside
the comparison component, where it belongs.

## Known limitations

- `BPP_TARGET = 0.55` is one constant for all content. A flat image and a noisy
  one have different crossovers, and the search compensates only by testing
  neighbouring resolutions.
- SSIM is luma-only. Chroma subsampling damage does not show up in the ranking.
- The resize grid is fixed and coarse below 0.3.
- WebP is capped at 16383 px per side; larger images silently fall back to JPEG.
- Animated images, ICC colour profiles and EXIF metadata other than orientation
  are not preserved. Metadata stripping is therefore implicit -- going through
  a NumPy array drops EXIF, including GPS -- which is good for privacy but is
  a side effect rather than a stated feature.
- Non-local means denoising costs ~0.85s at 0.8 MP and scales with pixel
  count, so a 12 MP image is slow. A faster path is needed there.
- The noise estimator underestimates by roughly a third; the auto-strength
  factor compensates, but a recalibration would be cleaner than a constant.

## Running things

```bash
python -m pip install -r requirements.txt

set PYTHONPATH=.                    # or export, on POSIX
python tests/test_compression.py -v
python tests/test_enhance.py -v
python benchmark.py
streamlit run app.py
```

Benchmark targets deliberately span the crossover: at 1200×1200, 25 KB is
0.14 bpp and 500 KB is 2.8 bpp. Testing only the high end hides half the
behaviour, which is how the resolution rule went unexercised for so long.
