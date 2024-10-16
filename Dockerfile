FROM vllm/vllm-openai:v0.6.3

ENV DO_NOT_TRACK=1

COPY --chmod=775 endpoints-entrypoint.sh entrypoint.sh

# Install dependencies for adding CUDA repository
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Add NVIDIA CUDA repository keyring
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb

# Update package lists after adding CUDA repository
RUN apt-get update

# Install CUDA toolkit (CUDA 12.1)
RUN apt-get install -y --no-install-recommends cuda-toolkit-12-1

# Set CUDA_HOME to the actual installation directory
ENV CUDA_HOME=/usr/local/cuda-12.1

# Update PATH environment variable
ENV PATH=$CUDA_HOME/bin:$PATH

# Create a symlink to /usr/local/cuda (some software expects CUDA here)
RUN ln -s $CUDA_HOME /usr/local/cuda

# Verify that nvcc is available (additional checks)
RUN find / -name nvcc
RUN python3 -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)"
RUN which nvcc
RUN nvcc --version

# Check CUDA version in PyTorch
RUN python3 -c "import torch; print(torch.version.cuda)"

# Clean up APT cache to reduce image size
RUN rm -rf /var/lib/apt/lists/*

# Install flash-attn using pip
RUN pip install flash-attn --no-build-isolation

ENTRYPOINT ["/bin/bash", "entrypoint.sh"]
CMD [""]
