# Dockerfile
# -----------------------------------------------------------------------------
# v0.2.0 — FROM-SCRATCH BUILD ON CUDA 12.6 + Ubuntu 24.04
# -----------------------------------------------------------------------------
# Why a full rebuild now (history):
#   v0.1.7-0.1.15 sat on top of styler00dollar/vsgan_tensorrt:minimal_no_avx512,
#   which is CUDA 13.0 + TensorRT 10.13. RunPod's pool of RTX 4090 hosts is on
#   driver 535-575 → CUDA runtime 12.x only. CUDA 13 requires libcuda 580+.
#   v0.1.14 tried `cuda-compat-13-0` to bridge, but NVIDIA hard-gates that
#   compat libcuda to Tesla-class GPUs (consumer RTX rejected). v0.1.15
#   reverted, leaving "driver insufficient" as the terminal error on consumer
#   pods. Path A (find a CUDA-12 styler tag): dead — only 4 tags exist on
#   Docker Hub, all CUDA 13 (verified 2026-05-28). Path B (build from scratch):
#   this Dockerfile.
#
# As a side-bonus the new base lets us:
#   - build ffmpeg with --enable-openssl → HTTPS source URLs work
#   - pin every layer to upstream tags (no third-party image drift)
#   - control which TensorRT / vstrt / VapourSynth versions ship together
#
# Build inputs (pinned via ARG so we can bump them safely):
#   - VS_VERSION=R72                 (matches styler's R72; R73 unreleased)
#   - VSMLRT_VERSION=15.13           (last CUDA-12 release; v15.14+ moved to CUDA 13)
#   - TENSORRT_VERSION=10.7.0.23     (TRT 10.7 ships for cuda-12.6)
#   - BESTSOURCE_TAG=R12             (matches the styler image build)
#   - FFMPEG_TAG=n7.1                (stable release with NVENC + OpenSSL)
#
# Build stages (single-stage for v0.2.0 — multi-stage optimization is Phase 2):
#   1. apt base deps (build toolchain + libs that ffmpeg/bs/lsmas link against)
#   2. pip cython for VapourSynth python bindings
#   3. zimg (VapourSynth required dep, not in apt for Noble)
#   4. VapourSynth R72 (autotools — `./autogen.sh && ./configure && make`)
#   5. TensorRT 10.7.0.23 tarball install (libs + headers + trtexec)
#   6. ffmpeg n7.1 with NVENC + OpenSSL + libx264/265/dav1d/opus
#   7. BestSource (bs) — meson build, used by run_rife.py for HTTP/HLS sources
#   8. L-SMASH-Works (lsmas) — kept as a fallback even though run_rife.py now
#      uses bs. /probe/lsmas_safe etc still exercise it for diagnostics.
#   9. vs-mlrt vstrt — CMake build, links against TensorRT 10.7 we installed in 5
#   10. vsmlrt.py Python wrapper + RIFE v4.6 ONNX model from upstream 7z assets
#   11. Verify vs.core has trt + bs namespaces (build fails if either missing)
#   12. Engine prebake (no-op without GPU at build time, kept for future CI GPU)
#   13. App code + nginx + nginx.conf + start.sh
#
# Hard constraints (verified before push):
#   - All 25 pytest unit tests still pass on host (no app code changed)
#   - Existing run_rife.py / server.py / pipeline_types.py / auth.py unchanged
#   - start.sh unchanged (auto-derive PUBLIC_BASE_URL + restart loop preserved)
#   - libvstrt.so symlinked into /usr/local/lib/vapoursynth/ so VS autoloads it
#   - vsmlrt.py installed at /usr/local/lib/python3.12/dist-packages/
#   - RIFE v4.6 model mirrored at /usr/local/lib/models/rife/ AND
#     /usr/local/lib/vapoursynth/models/rife/ (vsmlrt resolves dirname(.so)+/models)
#
# Image-size estimate: ~6-8 GB compressed. Bigger than v0.1.15 (5.5 GB) because
# TensorRT tarball alone is 4.4 GB uncompressed; we keep only the libs we need
# (libnvinfer*, libnvonnx*, libnvinfer_plugin*, trtexec) so net delta is modest.
# -----------------------------------------------------------------------------

# NOTE on base choice: cudnn-DEVEL (not -runtime) — first build attempt v0.2.0
# used -runtime, which omits nvcc + CUDA headers; ffmpeg's `--enable-cuda-nvcc`
# requires nvcc and vstrt's CMake build needs <cuda_runtime.h>. Devel ships both.
# Image is ~2 GB larger but that's a Phase-2 size optimization.
FROM nvidia/cuda:12.6.1-cudnn-devel-ubuntu24.04

ARG VS_VERSION=R72
ARG VSMLRT_VERSION=15.13
ARG TENSORRT_VERSION=10.7.0.23
ARG TENSORRT_CUDA_TAG=12.6
ARG BESTSOURCE_TAG=R12
ARG FFMPEG_TAG=n7.1

# Avoid tzdata interactive prompt during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Make ldconfig pick up /usr/local/lib at every layer (zimg, vapoursynth,
# bestsource, lsmas, vstrt all install here).
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/x86_64-linux-gnu:/usr/local/cuda/lib64

WORKDIR /app

# -----------------------------------------------------------------------------
# 1. System dependencies (build toolchain + media libs ffmpeg/bs/lsmas link to)
# -----------------------------------------------------------------------------
# Notes:
#   - Python 3.12 is the default in Noble; python3-pip + python3-dev cover us
#   - autoconf/automake/libtool needed for VapourSynth R72 (uses autotools)
#   - meson/ninja-build needed for bestsource + L-SMASH-Works
#   - nasm/yasm needed by ffmpeg's libx264/libx265 builds
#   - libssl-dev for ffmpeg --enable-openssl (fixes the HTTPS-source gotcha)
#   - libxxhash-dev needed by bestsource
#   - p7zip-full for extracting upstream vs-mlrt 7z assets
#   - nginx from apt — Noble ships a recent enough nginx for our needs
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget git pkg-config \
        build-essential cmake ninja-build meson \
        autoconf automake libtool \
        nasm yasm \
        python3 python3-dev python3-pip python3-venv \
        libssl-dev \
        libx264-dev libx265-dev libnuma-dev libvpx-dev libdav1d-dev \
        libopus-dev libfdk-aac-dev libmp3lame-dev libvorbis-dev \
        zlib1g-dev libxml2-dev libfreetype6-dev libfontconfig1-dev \
        libxxhash-dev \
        p7zip-full \
        nginx \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 2. Cython for VapourSynth Python bindings
# -----------------------------------------------------------------------------
# Noble enforces PEP-668; --break-system-packages keeps things simple for an
# image whose entire purpose is one Python app. Cython 3.1+ is required by
# VapourSynth R72 (per upstream docs).
# -----------------------------------------------------------------------------
RUN pip install --no-cache-dir --break-system-packages "cython>=3.1"

# -----------------------------------------------------------------------------
# 3. zimg (VapourSynth dependency, not in Noble apt)
# -----------------------------------------------------------------------------
RUN git clone --depth 1 -b release-3.0.5 https://github.com/sekrit-twc/zimg.git /tmp/zimg \
    && cd /tmp/zimg \
    && ./autogen.sh \
    && ./configure --prefix=/usr/local --disable-static --enable-shared \
    && make -j"$(nproc)" && make install \
    && ldconfig \
    && rm -rf /tmp/zimg

# -----------------------------------------------------------------------------
# 4. VapourSynth R72 from source
# -----------------------------------------------------------------------------
# R72 uses autotools (autogen.sh → configure → make). The styler upstream
# Dockerfile uses the same recipe. After install, ldconfig picks up
# libvapoursynth.so from /usr/local/lib.
#
# Python bindings install to /usr/local/lib/python3.12/dist-packages/vapoursynth.*
# automatically because `make install` runs `python3 setup.py install` for
# the Python subdir. The plugin autoload path is
# /usr/local/lib/vapoursynth/, which `core.std` etc populate at runtime
# (matches our existing v0.1.15 symlink target).
# -----------------------------------------------------------------------------
RUN git clone --depth 1 -b ${VS_VERSION} https://github.com/vapoursynth/vapoursynth.git /tmp/vs \
    && cd /tmp/vs \
    && ./autogen.sh \
    && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" && make install \
    && ldconfig \
    && rm -rf /tmp/vs

# -----------------------------------------------------------------------------
# 4b. Symlink VapourSynth Python binding into dist-packages
# -----------------------------------------------------------------------------
# VapourSynth's autotools install lands the Python module in
# /usr/local/lib/python3.12/site-packages/, but Ubuntu 24.04's Python only adds
# /usr/local/lib/python3.12/dist-packages/ to sys.path by default. Without this
# bridge, `import vapoursynth` fails even though the .so is on disk — which is
# exactly what blew up the v0.2.2 verification step.
#
# Use symlinks (ln -sf) not copies — preserves the upstream install location
# and keeps the layer small. The trailing python3 -c verification fails the
# build immediately if the symlink doesn't resolve, giving fast-fail instead of
# waiting until the step-11 namespace assertion.
# -----------------------------------------------------------------------------
RUN mkdir -p /usr/local/lib/python3.12/dist-packages \
    && for f in /usr/local/lib/python3.12/site-packages/vapoursynth*; do \
           if [ -e "$f" ]; then \
               ln -sf "$f" "/usr/local/lib/python3.12/dist-packages/$(basename $f)"; \
           fi; \
       done \
    && python3 -c "import vapoursynth as vs; print('VS imported OK, core:', vs.core)"

# -----------------------------------------------------------------------------
# 5. TensorRT 10.7.0.23 (tarball install — apt repo is broken for CUDA 12)
# -----------------------------------------------------------------------------
# Why the tarball: NVIDIA's apt repo for libnvinfer10 has unmet-dep bugs on
# Ubuntu 24.04 with CUDA 12 (see NVIDIA/TensorRT issues #4545, #4593). The
# tarball is the documented escape hatch.
#
# We keep only what's needed at runtime + build time:
#   - lib/libnvinfer*.so*       (TensorRT engine + plugins)
#   - lib/libnvonnx*.so*        (ONNX parser, used during engine compile)
#   - include/                  (headers, needed for vstrt CMake build)
#   - bin/trtexec               (handy for debugging engine compile)
#
# The Python wheels in the tarball (python/*.whl) we DON'T install — vs-mlrt
# uses the C++ libs directly via the vstrt plugin, not the Python TRT bindings.
# Saves ~1 GB.
# -----------------------------------------------------------------------------
RUN curl -fSL \
        "https://developer.download.nvidia.com/compute/machine-learning/tensorrt/${TENSORRT_VERSION%.*}/tars/TensorRT-${TENSORRT_VERSION}.Linux.x86_64-gnu.cuda-${TENSORRT_CUDA_TAG}.tar.gz" \
        -o /tmp/TensorRT.tar.gz \
    && mkdir -p /opt/tensorrt \
    && tar -xzf /tmp/TensorRT.tar.gz -C /opt/tensorrt --strip-components=1 \
    # libnvinfer* covers nvinfer, nvinfer_plugin, nvinfer_dispatch, nvinfer_lean,
    # nvinfer_vc_plugin. libnvonnxparser is the .so name in TRT 10.x.
    # libnvparsers/libnvcaffe_parser were removed after TRT 9.x.
    && cp -d /opt/tensorrt/lib/libnvinfer*.so* /usr/local/lib/ \
    && cp -d /opt/tensorrt/lib/libnvonnxparser*.so* /usr/local/lib/ 2>/dev/null || true \
    && cp -r /opt/tensorrt/include/* /usr/local/include/ \
    # v0.2.5: install the REAL trtexec binary at /usr/local/bin/trtexec.real,
    # then write a wrapper at /usr/local/bin/trtexec that explicitly sets
    # LD_LIBRARY_PATH before exec'ing the real binary. Why the wrapper:
    # vsmlrt.py:2192 launches trtexec via subprocess.run(args, env={...})
    # where the env dict only contains TRTEXEC_LOG_FILE + CUDA_MODULE_LOADING.
    # That STRIPS LD_LIBRARY_PATH from the child env, so the Dockerfile-level
    # `ENV LD_LIBRARY_PATH=...` (set below) is silently invisible to trtexec
    # when invoked from RIFE engine compilation. The wrapper script re-injects
    # the lib paths so trtexec can resolve libnvinfer*, libcudnn*, libcudart*
    # etc. even when the calling process strips env. ldconfig (run below) is a
    # belt-and-suspenders fallback for the same problem.
    && cp /opt/tensorrt/bin/trtexec /usr/local/bin/trtexec.real 2>/dev/null || true \
    && printf '#!/bin/bash\nexport LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/opt/tensorrt/lib:${LD_LIBRARY_PATH}\nexec /usr/local/bin/trtexec.real "$@"\n' > /usr/local/bin/trtexec \
    && chmod +x /usr/local/bin/trtexec \
    # vsmlrt.py hardcodes trtexec_path = os.path.join(plugins_path, "vsmlrt-cuda", "trtexec")
    # where plugins_path is the directory containing libvstrt.so. The vstrt plugin
    # lives at /usr/local/lib/vapoursynth/libvstrt.so, so vsmlrt expects
    # /usr/local/lib/vapoursynth/vsmlrt-cuda/trtexec. We symlink /usr/local/bin/trtexec
    # (the wrapper) in. (Discovered v0.2.3 pod test: vsmlrt raised FileNotFoundError
    # because the expected path did not exist even though trtexec was on PATH.)
    && mkdir -p /usr/local/lib/vapoursynth/vsmlrt-cuda \
    && ln -sf /usr/local/bin/trtexec /usr/local/lib/vapoursynth/vsmlrt-cuda/trtexec \
    && rm -rf /tmp/TensorRT.tar.gz \
    # v0.2.5: KEEP /opt/tensorrt/lib around (we used to rm -rf it). The wrapper's
    # LD_LIBRARY_PATH includes /opt/tensorrt/lib as the last fallback in case any
    # TRT plugin lib didn't get cp'd into /usr/local/lib (e.g. libnvinfer_lean
    # extras). Cost: ~1.5 GB extra image size; acceptable for a diagnostic
    # version. Strip it back out in v0.2.6 if the wrapper proves unnecessary.
    && rm -rf /opt/tensorrt/python /opt/tensorrt/data /opt/tensorrt/doc /opt/tensorrt/samples /opt/tensorrt/targets \
    && ldconfig

# v0.2.5: explicit LD_LIBRARY_PATH for any non-vsmlrt caller (e.g. our own
# /probe/trtexec endpoint runs trtexec via subprocess.run with full os.environ
# inherited, so this ENV is what it sees). vsmlrt strips env when calling
# trtexec — see wrapper script comment above for the real fix.
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/opt/tensorrt/lib

# -----------------------------------------------------------------------------
# 6. ffmpeg n7.1 with NVENC + NVDEC + OpenSSL
# -----------------------------------------------------------------------------
# Critical flags:
#   --enable-openssl   → HTTPS source URLs work (was the big v0.1.x gotcha)
#   --enable-nvenc/dec → h264_nvenc for the encode side of run_rife.py
#   --enable-libnpp    → CUDA scaling fast-path
#   --enable-libdav1d  → AV1 sources
#
# nv-codec-headers provides cuviddec.h etc. We grab them from FFmpeg's
# preferred shallow tag (matches FFmpeg n7.1 ABI).
# -----------------------------------------------------------------------------
RUN git clone --depth 1 -b n12.2.72.0 https://github.com/FFmpeg/nv-codec-headers.git /tmp/nvh \
    && cd /tmp/nvh && make install \
    && rm -rf /tmp/nvh

RUN git clone --depth 1 -b ${FFMPEG_TAG} https://github.com/FFmpeg/FFmpeg.git /tmp/ffmpeg \
    && cd /tmp/ffmpeg \
    && ./configure \
        --prefix=/usr/local \
        --enable-gpl --enable-nonfree \
        --enable-openssl \
        --enable-libx264 --enable-libx265 --enable-libdav1d --enable-libopus \
        --enable-libfdk-aac --enable-libmp3lame --enable-libvorbis \
        --enable-cuda-nvcc --enable-nvenc --enable-nvdec --enable-libnpp \
        --extra-cflags="-I/usr/local/cuda/include -I/usr/local/include" \
        --extra-ldflags="-L/usr/local/cuda/lib64 -L/usr/local/lib" \
    && make -j"$(nproc)" && make install \
    && ldconfig \
    && rm -rf /tmp/ffmpeg \
    && ffmpeg -version | head -3

# -----------------------------------------------------------------------------
# 7. BestSource (bs.VideoSource) — needed for HTTP/HLS source decoding
# -----------------------------------------------------------------------------
# We use styler00dollar's fork to match the API surface used by run_rife.py
# (cachemode=0 etc). meson installs the .so into /usr/local/lib/vapoursynth/,
# so VS autoloads it on import. R12 matches the styler image.
# -----------------------------------------------------------------------------
RUN git clone --depth 1 -b ${BESTSOURCE_TAG} --recurse-submodules \
        https://github.com/vapoursynth/bestsource.git /tmp/bs \
    && cd /tmp/bs \
    && CFLAGS=-fPIC meson setup -Denable_plugin=true build \
    && CFLAGS=-fPIC ninja -C build \
    && ninja -C build install \
    && ldconfig \
    && rm -rf /tmp/bs

# -----------------------------------------------------------------------------
# 8. L-SMASH-Works (lsmas) intentionally OMITTED in v0.2.2:
# -----------------------------------------------------------------------------
# 1. run_rife.py uses bs.VideoSource (BestSource), not lsmas.LWLibavSource
# 2. AkarinVS/L-SMASH-Works' VapourSynth meson build requires liblsmash
#    pre-installed; the bundled submodule isn't built by the VS subproject
# 3. Dropping it removes a 5-min build step + simplifies the image
# 4. /probe/lsmas_* diagnostic endpoints will return "namespace not found"
#    at runtime — acceptable since the lsmas-can't-do-HTTP bug was already
#    diagnosed in v0.1.10
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 9. vs-mlrt vstrt plugin (the TensorRT VapourSynth bridge)
# -----------------------------------------------------------------------------
# CMake build. USE_NVINFER_PLUGIN=ON pulls in TensorRT's extra ops which RIFE
# v4.6 doesn't strictly need but is harmless and matches the styler build.
# Installs libvstrt.so to /usr/local/lib/. We then symlink to
# /usr/local/lib/vapoursynth/ so VS autoload picks it up (same fix v0.1.5).
# -----------------------------------------------------------------------------
RUN git clone --depth 1 -b v${VSMLRT_VERSION} \
        https://github.com/AmusementClub/vs-mlrt.git /tmp/vsmlrt \
    && cd /tmp/vsmlrt/vstrt \
    && cmake -B build -G Ninja \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DUSE_NVINFER_PLUGIN=ON \
        -DVAPOURSYNTH_INCLUDE_DIRECTORY=/usr/local/include/vapoursynth \
    && cmake --build build \
    && cmake --install build \
    && mkdir -p /usr/local/lib/vapoursynth \
    # The CMake install rule lands libvstrt.so in /usr/local/lib (we forced
    # LIBDIR=lib). VS autoloads .so files from /usr/local/lib/vapoursynth/,
    # so symlink. If libvstrt didn't land in /usr/local/lib (e.g. ended up in
    # the build/ tree or lib/x86_64-linux-gnu), search and copy as a fallback.
    && if [ -f /usr/local/lib/libvstrt.so ]; then \
           ln -sf /usr/local/lib/libvstrt.so /usr/local/lib/vapoursynth/libvstrt.so; \
       else \
           VSTRT_SO="$(find /tmp/vsmlrt/vstrt/build /usr/local -maxdepth 4 -name 'libvstrt.so*' 2>/dev/null | head -1)"; \
           if [ -n "$VSTRT_SO" ]; then \
               cp "$VSTRT_SO" /usr/local/lib/libvstrt.so; \
               ln -sf /usr/local/lib/libvstrt.so /usr/local/lib/vapoursynth/libvstrt.so; \
           else \
               echo "ERROR: libvstrt.so not found after build" >&2; exit 1; \
           fi; \
       fi \
    && ls -la /usr/local/lib/libvstrt.so /usr/local/lib/vapoursynth/libvstrt.so \
    && ldconfig \
    && rm -rf /tmp/vsmlrt

# -----------------------------------------------------------------------------
# 10. vsmlrt.py Python module + RIFE v4.6 ONNX model
# -----------------------------------------------------------------------------
# Same as v0.1.x — download scripts.v15.13.7z and models.v15.13.7z from the
# upstream release page. vsmlrt.py resolves the model dir from where the .so
# loaded, so we mirror to both /usr/local/lib/models/ and
# /usr/local/lib/vapoursynth/models/ as belt-and-suspenders.
# -----------------------------------------------------------------------------
RUN curl -L -o /tmp/vsmlrt-scripts.7z \
        https://github.com/AmusementClub/vs-mlrt/releases/download/v${VSMLRT_VERSION}/scripts.v${VSMLRT_VERSION}.7z \
    && 7z x /tmp/vsmlrt-scripts.7z -o/tmp/vsmlrt-scripts \
    && cp /tmp/vsmlrt-scripts/vsmlrt.py /usr/local/lib/python3.12/dist-packages/ \
    && rm -rf /tmp/vsmlrt-scripts /tmp/vsmlrt-scripts.7z

RUN mkdir -p /usr/local/lib/models /usr/local/lib/vapoursynth/models \
    && curl -L -o /tmp/vsmlrt-models.7z \
        https://github.com/AmusementClub/vs-mlrt/releases/download/v${VSMLRT_VERSION}/models.v${VSMLRT_VERSION}.7z \
    && 7z x /tmp/vsmlrt-models.7z -o/tmp/vsmlrt-models -y \
    && cp -r /tmp/vsmlrt-models/models/rife /usr/local/lib/models/ \
    && cp -r /tmp/vsmlrt-models/models/rife /usr/local/lib/vapoursynth/models/ \
    && rm -rf /tmp/vsmlrt-models /tmp/vsmlrt-models.7z

# -----------------------------------------------------------------------------
# 11. Verify VS namespaces include trt + bs (FAIL the build if missing)
# -----------------------------------------------------------------------------
# This is the critical check. If either plugin failed to autoload, the build
# fails here loudly instead of producing an image that 502s on first /process.
# We DO NOT call any GPU primitives — that requires runtime CUDA which CI
# doesn't have. Just verify the .so plugins are importable as namespaces.
# -----------------------------------------------------------------------------
RUN python3 -c "import vapoursynth as vs; \
ns = sorted([p for p in dir(vs.core) if not p.startswith('_')]); \
print('VS namespaces:', ns); \
assert 'trt' in ns, 'trt namespace missing — vstrt plugin failed to autoload'; \
assert 'bs' in ns, 'bs namespace missing — bestsource plugin failed to autoload'; \
print('OK: vstrt + bestsource autoloaded')"

# -----------------------------------------------------------------------------
# 12. Engine prebake (no-op without GPU at build time)
# -----------------------------------------------------------------------------
RUN mkdir -p /root/.cache/vs-mlrt
COPY trt_engine_builder.py /tmp/
RUN python3 /tmp/trt_engine_builder.py \
    || echo "Engine bake skipped (no GPU at build time — compiles on first /process)"

# -----------------------------------------------------------------------------
# 13. Python deps + app code + nginx config
# -----------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY server.py auth.py pipeline.py pipeline_types.py run_rife.py idle_watcher.py start.sh ./
COPY nginx.conf /etc/nginx/nginx.conf

# RunPod exposes ONE port per pod — nginx terminates :8080, proxies / to
# uvicorn:8001 and serves /hls/* directly. /var/hls is the on-disk segment dir.
EXPOSE 8080
ENV HLS_SERVE_DIR=/var/hls

CMD ["./start.sh"]
