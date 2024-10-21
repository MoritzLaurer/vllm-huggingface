FROM vllm/vllm-openai:v0.6.3

ENV DO_NOT_TRACK=1

COPY --chmod=775 endpoints-entrypoint.sh entrypoint.sh

# Check CUDA and pytorch version to simplify debugging
RUN python3 -c "import torch; print(torch.version.cuda); print(torch.__version__)"

# Install flash-attn using pip
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

ENTRYPOINT ["/bin/bash", "entrypoint.sh"]
CMD [""]
