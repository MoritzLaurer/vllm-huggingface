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

# Install CUDA toolkit matching your CUDA version (CUDA 12.1)
RUN apt-get install -y --no-install-recommends cuda-toolkit-12-1

# Set CUDA_HOME to /usr since CUDA 12.1 installs to /usr
ENV CUDA_HOME=/usr

# Update PATH environment variable
ENV PATH=$CUDA_HOME/bin:$PATH

# Verify that nvcc is available (additional checks)
RUN which nvcc
RUN nvcc --version

# Clean up APT cache to reduce image size
RUN rm -rf /var/lib/apt/lists/*

# Install flash-attn using pip
RUN pip install flash-attn --no-build-isolation

# Check CUDA version
RUN python -c "import torch; print(torch.version.cuda)"

ENTRYPOINT ["/bin/bash", "entrypoint.sh"]
CMD [""]
