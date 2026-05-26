# castbooster-cloud-worker

Cloud worker for castbooster.ai's Pillar 3.6 MVP. Runs in a RunPod RTX 4090 pod;
fetches source HLS, runs RIFE v4.6 TensorRT FP16 inference for 60fps temporal
interpolation, re-encodes via NVENC, serves output as HLS on `*.proxy.runpod.net`.

Single-user MVP for project owner. Not a production multi-tenant service.

See `docs/superpowers/specs/2026-05-24-pillar-3.6-cloud-vfi-design.md` in the
main castbooster repo for full design.
