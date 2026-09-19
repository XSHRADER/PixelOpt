# Sample inputs

Eight images, each built to test one specific behaviour, so a wrong result is
easy to spot. They are generated rather than downloaded: reproducible, but
synthetic, so compare the quality numbers with each other rather than treating
them as absolute.

```bash
python samples/make_samples.py      # builds samples/inputs/
python samples/run_samples.py       # writes samples/outputs/ + contact_sheet.png
```

## What each one tests

| file | what it tests | try in the app | what you should see |
|---|---|---|---|
| `01_noisy_photo.jpg` | denoise pays for itself | 150 KB, then switch Enhancement between **none** and **Auto** | "Measured noise 12.4". Fidelity **0.601 → 0.993** at the same budget. Zoom the sky to 2:1 — the grain is gone |
| `02_clean_landscape.jpg` | auto leaves clean images alone | 60 KB, Enhancement **Auto** | "clean enough to leave alone", no "Enhanced" line, drift 1.000 |
| `03_screenshot_text.png` | lossless wins on flat art | 100 KB, Encoder **Auto** | Encoder **PNG**, fidelity **1.0000**, text razor-sharp at 4:1 |
| `04_logo_transparent.png` | transparency handling | 30 KB | Yellow transparency warning, encoder PNG, background **white** (not black) |
| `05_portrait_exif_rotated.jpg` | EXIF orientation | 80 KB | Output **900 × 1400** (tall), "THIS SIDE UP" bar at the top |
| `06_large_12mp_detail.jpg` | resolution rule at low bpp | 200 KB | Caption shows ~0.13 bpp; dimensions **1600 × 1200** (resize 0.40×). Takes ~15 s |
| `07_faded_scan.png` | scan preset | 250 KB, Enhancement **none**, then **scan** | Fidelity **0.833 → 0.993**; paper whiter, text near-black |
| `08_blue_cast_photo.jpg` | white balance | 80 KB, open *Fine-tune*, toggle **White balance** | Blue cast removed — see the caveat below |

## Measured on the last run

| case | input | target | output | encoder | dimensions | fidelity | drift | enhancement |
|---|---|---|---|---|---|---|---|---|
| noisy, plain | 1116.1 KB | 150 KB | 147.4 KB | JPEG | 1600×1067 | 0.601 | 1.000 | – |
| noisy, auto | 1116.1 KB | 150 KB | 136.4 KB | JPEG | 1600×1067 | **0.993** | 0.335 | denoise 10 |
| clean, auto | 106.2 KB | 60 KB | 39.6 KB | WebP | 1600×1067 | 0.996 | 1.000 | – |
| screenshot | 54.3 KB | 100 KB | 54.3 KB | PNG | 1280×800 | 1.000 | 1.000 | – |
| logo | 12.5 KB | 30 KB | 10.3 KB | PNG | 900×900 | 1.000 | 1.000 | – |
| portrait | 83.8 KB | 80 KB | 76.9 KB | JPEG | **900×1400** | 0.999 | 1.000 | – |
| 12 MP detail | 1939.1 KB | 200 KB | 199.7 KB | JPEG | **1600×1200** | 0.970 | 1.000 | – |
| scan, plain | 4479.6 KB | 250 KB | 245.9 KB | JPEG | 1240×1754 | 0.833 | 1.000 | – |
| scan, preset | 4479.6 KB | 250 KB | 243.2 KB | WebP | 1240×1754 | **0.993** | 0.745 | denoise, levels, contrast, sharpen |
| blue cast, WB | 100.2 KB | 80 KB | 74.6 KB | JPEG | 1600×1067 | 0.998 | 0.996 | white balance |

Every output landed at or under its target.

Checks the numbers alone cannot make, verified separately:

- **Logo background** is white: corner pixel `(255, 255, 255)`.
- **Portrait** is upright: the top band averages brightness 54 (the dark label
  bar) against 178 just below it.
- **Scan preset** contrast rose from 144 to 231 levels; ink went from 77 to 2.
- **White balance** pulled the channel means from R 87 / G 127 / B 175 to
  129 / 129 / 129.

## Things worth knowing

**White balance over-corrects this sample.** Gray-world assumes the average
scene colour is neutral grey, so a picture dominated by blue sky gets its sky
dragged toward lavender-grey — visible in the last row of the contact sheet.
It works on genuinely cast images and fails on scenes that are *supposed* to
be mostly one colour. This is also why auto mode never turns it on. Note the
drift of 0.996: SSIM is computed on brightness only, so it barely registers a
colour shift. Judge this one by eye.

**Drift is not an error.** 0.335 on the noisy photo is the denoiser removing a
lot of grain; 0.745 on the scan is levels and contrast doing their job.

**Fidelity measures the encoder, not the enhancement.** It is SSIM against the
enhanced reference. That is why the plain scan scores 0.833 — the encoder is
spending its budget on paper texture and scanner noise.

**The 12 MP case is slow**, about 17 s. Around 6.5 s of that is the lossless
PNG candidate being encoded at full resolution, where a 6.4 MB PNG could never
fit a 200 KB budget.

**Judge compression at 1:1 or closer.** At fit-to-screen, most of these look
identical before and after. In the app, use the 1:1 / 2:1 / 4:1 buttons; the
contact sheet shows 2× crops for the same reason.
