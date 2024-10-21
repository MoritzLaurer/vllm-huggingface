from huggingface_hub import create_inference_endpoint
import os
from dotenv import load_dotenv
import re


VLLM_HF_IMAGE_URL = "moritzlaurer/vllm-huggingface:latest"
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"  #"OpenGVLab/InternVL2-4B" #"microsoft/Phi-3.5-vision-instruct"

def create_compatible_endpoint_name(model_id: str) -> str:
    part = model_id.split('/')[-1]
    part_lower = part.lower()
    cleaned = re.sub(r'[^a-z0-9\-]', '-', part_lower)
    trimmed = cleaned[:32]
    return trimmed


if __name__ == "__main__":
    load_dotenv()
    repo_id = MODEL_ID
    env_vars = {
        "DISABLE_SLIDING_WINDOW": "false",
        "MAX_MODEL_LEN": "4096",
        #"MAX_NUM_BATCHED_TOKENS": "8192",
        "DTYPE": "bfloat16",
        "GPU_MEMORY_UTILIZATION": "0.98",
        #"QUANTIZATION": "fp8",
        #"USE_V2_BLOCK_MANAGER": "true",
        "VLLM_ATTENTION_BACKEND": "FLASHINFER",
        "TRUST_REMOTE_CODE": "true",
    }

    endpoint = create_inference_endpoint(
        name=create_compatible_endpoint_name(MODEL_ID),
        repository=repo_id,
        framework="pytorch",
        task="custom",
        accelerator="gpu",
        vendor="aws",
        region="us-east-1",
        type="protected",
        instance_size="x1",
        instance_type="nvidia-a10g",
        min_replica=0,
        max_replica=1,
        scale_to_zero_timeout=30,
        custom_image={
            "health_route": "/health",
            "env": env_vars,
            "url": VLLM_HF_IMAGE_URL,
        },
        token=os.getenv("HF_TOKEN"),
    )
    
    print(f"Go to https://ui.endpoints.huggingface.co/{endpoint.namespace}/endpoints/{endpoint.name} to see the endpoint status.")
