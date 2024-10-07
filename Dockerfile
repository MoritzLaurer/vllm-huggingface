FROM vllm/vllm-openai:v0.6.2

ENV VLLM_COMMIT=7193774b1ff8603ad5bf4598e5efba0d9a39b436
ENV VLLM_VERSION=0.6.2
ENV DO_NOT_TRACK=1

COPY --chmod=775 endpoints-entrypoint.sh entrypoint.sh

RUN pip uninstall -y vllm && \
    pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/${VLLM_COMMIT}/vllm-${VLLM_VERSION}-cp38-abi3-manylinux1_x86_64.whl

ENTRYPOINT ["/bin/bash", "entrypoint.sh"]
CMD [""]
