FROM vllm/vllm-openai:v0.6.3.post1

ENV DO_NOT_TRACK=1

COPY --chmod=775 endpoints-entrypoint.sh entrypoint.sh

# Check CUDA and pytorch version for debugging
RUN python3 -c "import torch; print(torch.version.cuda); print(torch.__version__)"

# Install flashinfer. much easier to install than flash-attn
RUN pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

ENTRYPOINT ["/bin/bash", "entrypoint.sh"]
CMD [""]
