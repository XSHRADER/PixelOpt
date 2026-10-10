"""Name what is in an image, remove one thing, predict what was behind it.

Kept in its own subpackage because it needs libraries the rest of PixelOpt
must never import: torch, transformers and diffusers arrive only with the
optional `vision` extra. Inside this package those imports happen only in the
functions that load or run a model, so every pure helper -- parsing, masks,
cropping, compositing, scoring -- works, and is tested, without them.
"""
