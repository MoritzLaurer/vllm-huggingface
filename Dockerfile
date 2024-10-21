FROM vllm/vllm-openai:v0.6.3.post1

ENV DO_NOT_TRACK=1

COPY --chmod=775 endpoints-entrypoint.sh entrypoint.sh

# Install dependencies for adding CUDA repository
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg2 \
    ca-certificates \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-toolkit-12-1 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV CUDA_HOME=/usr/local/cuda-12.1
ENV PATH=$CUDA_HOME/bin:$PATH

# Create a symlink to /usr/local/cuda (some software expects CUDA here)
RUN ln -s $CUDA_HOME /usr/local/cuda

# Verify that nvcc is available (additional checks)
RUN find / -name nvcc && \
    python3 -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)" && \
    which nvcc && \
    nvcc --version

# Check CUDA and pytorch version to simplify debugging
RUN python3 -c "import torch; print(torch.version.cuda); print(torch.__version__)"

# Install flash-attn using pip
RUN pip install flash-attn --no-build-isolation

ENTRYPOINT ["/bin/bash", "entrypoint.sh"]
CMD [""]
