# DiMA – Diffusion on Language Model Encodings for Protein Sequence Generation
#
# Base: CUDA 12.1 devel image for full CUDA headers (required by PyTorch/ninja).
# PyTorch 2.5.1 is built for CUDA 12.1, so they must match.
# openfold is installed as pure Python (no CUDA kernel compilation): DiMA only
# uses openfold's Python modules via cheap-proteins (residue_constants, etc.).

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# ── system packages ────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget git curl ca-certificates \
        build-essential gcc g++ \
        libssl-dev zlib1g-dev \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

# ── miniconda ─────────────────────────────────────────────────────────────────
ENV CONDA_DIR=/opt/conda
ENV PATH=/opt/conda/bin:$PATH

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && conda tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels \
        --channel https://repo.anaconda.com/pkgs/r \
    && conda clean -afy

# ── conda environment (pytorch + core scientific stack via conda) ──────────────
# We create the env manually rather than from environment.yaml so we can skip
# the git-dep lines (which break conda's pip isolation) and handle them below.
RUN conda create -n dima_env python=3.10 pip=23.0 -y && \
    conda install -n dima_env \
        -c pytorch -c nvidia -c conda-forge \
        pytorch=2.5.1 \
        pytorch-cuda=12.1 \
        torchvision=0.20.1 \
        torchmetrics=1.6.0 \
        "gxx_linux-64>=9,<=12" \
        ninja \
        matplotlib=3.7.1 \
        "numpy=1.24.3" \
        "pandas=2.0.1" \
        scipy \
        jupyter \
        ipython \
        ipykernel \
        -y \
    && conda clean -afy

# ── pip packages (non-git) ────────────────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt

RUN conda run -n dima_env pip install --no-cache-dir setuptools && \
    conda run -n dima_env pip install --no-cache-dir \
        accelerate==0.29.3 \
        "argparse==1.4.0" \
        ml-collections==0.1.1 \
        nltk==3.8.1 \
        scikit-image==0.21.0 \
        scikit-learn==1.3.2 \
        seaborn==0.12.2 \
        timm==0.9.16 \
        tqdm==4.66.3 \
        transformers==4.40.0 \
        wandb \
        "datasets==3.6.0" \
        "huggingface_hub==0.24.0" \
        evaluate \
        biopython==1.81 \
        biotite==0.37.0 \
        sentencepiece==0.1.99 \
        "hydra-core>=1.3" \
        lightning \
        einops \
        numba \
        dm-tree \
        modelcif \
        omegaconf \
        h5py \
        lmdb \
        safetensors \
        py3Dmol \
        pyarrow \
        plotly \
        ninja

# ── git-based deps ────────────────────────────────────────────────────────────
# Install via setup.py / pip inside the conda env so torch is visible.
# Order: esm → evo → cheap-proteins → openfold

# ESM (pinned commit)
RUN conda run -n dima_env pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/esm.git@d7b3331f41442ed4ffde70cb95bdd48cabcec2e9"

# EVO
RUN conda run -n dima_env pip install --no-cache-dir \
    "git+https://github.com/amyxlu/evo.git"

# cheap-proteins
RUN conda run -n dima_env pip install --no-cache-dir \
    "git+https://github.com/MeshchaninovViacheslav/cheap-proteins.git"

# openfold – build with CUDA kernels. CUDA 12.x dropped compute_37 (Kepler)
# and compute_52 (Maxwell). openfold's setup.py hardcodes these in the default
# compute_capabilities set; when docker build runs without a GPU the fallback
# includes them. We patch setup.py to replace the default set with sm_61+.
ENV CUDA_HOME=/usr/local/cuda
RUN git clone --depth 1 https://github.com/amyxlu/openfold.git /tmp/openfold_src && \
    sed -i 's/compute_capabilities = set(\[$/compute_capabilities = set([  # patched/' \
        /tmp/openfold_src/setup.py && \
    sed -i 's/(3, 7), # K80, e.g./# (3, 7) removed – unsupported by CUDA 12.x/' \
        /tmp/openfold_src/setup.py && \
    sed -i 's/(5, 2), # Titan X/# (5, 2) removed – unsupported by CUDA 12.x/' \
        /tmp/openfold_src/setup.py && \
    conda run -n dima_env bash -c \
        "cd /tmp/openfold_src && \
         CUDA_HOME=/usr/local/cuda MAX_JOBS=4 python setup.py install" && \
    rm -rf /tmp/openfold_src

# ── project source ─────────────────────────────────────────────────────────────
WORKDIR /workspace/DiMA
COPY . .

# Ensure project root is on the Python path inside the env
ENV PYTHONPATH=/workspace/DiMA
# Tell DiMA where the project root is (used by config.yaml via oc.env)
ENV PROJECT_ROOT=/workspace/DiMA
# Make PyTorch shared libs visible to openfold's attn_core_inplace_cuda.so
ENV LD_LIBRARY_PATH=/opt/conda/envs/dima_env/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH}

# ── cache dirs (mounted as volumes at runtime) ─────────────────────────────────
ENV HF_HOME=/workspace/DiMA/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/DiMA/.cache/huggingface/hub
RUN mkdir -p /workspace/DiMA/.cache/huggingface \
             /workspace/DiMA/checkpoints

# ── default runtime settings ──────────────────────────────────────────────────
ENV WANDB_MODE=offline

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "dima_env"]
CMD ["python", "example_simple.py"]
