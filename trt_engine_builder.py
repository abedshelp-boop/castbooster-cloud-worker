# trt_engine_builder.py
"""Bake the RIFE v4.6 TRT FP16 engine for RTX 4090 (sm_89) at docker build time.

If no GPU is present at build time, this script exits cleanly — the engine
will be compiled on the first /process call (~5-10 min slower cold start).
"""
import sys

try:
    import vapoursynth as vs
    from vsmlrt import RIFE, Backend
except ImportError as e:
    print(f"vsmlrt not available, skipping engine bake: {e}")
    sys.exit(0)

try:
    core = vs.core
    blank = core.std.BlankClip(width=1920, height=1080, format=vs.RGBS, length=2, fpsnum=30)
    out = RIFE(
        blank,
        multi=2,
        model=46,
        backend=Backend.TRT(fp16=True, num_streams=2),
    )
    # Force a single frame realization to trigger TRT engine compile
    out.get_frame(0)
    print("RIFE v4.6 TRT FP16 engine baked successfully")
except Exception as e:
    print(f"Engine bake failed (will compile at runtime): {e}")
    sys.exit(0)
