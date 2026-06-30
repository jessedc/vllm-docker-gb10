services:
  vllm:
    # you can define special tag instead of latest to have stable version
    #image: ghcr.io/aeon-7/aeon-vllm-ultimate:v0.22.1-pr44389-spark
    image: ghcr.io/aeon-7/aeon-vllm-ultimate:latest
    container_name: aeon-ultimate-xs-tq
    restart: unless-stopped
    network_mode: host
    ipc: host
    ulimits:
      memlock: -1

    environment:
      # ─── Core vLLM ───
      - VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

      # ─── GB10 / sm_121a runtime ───
      - TORCH_CUDA_ARCH_LIST=12.1a
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      - TORCH_MATMUL_PRECISION=high
      - NVIDIA_FORWARD_COMPAT=1
      - NVIDIA_DISABLE_REQUIRE=1

      # Required by PR #40191 for sm_121a-only builds — without this,
      # vllm._C_stable_libtorch.abi3.so fails to import.
      - ENABLE_NVFP4_SM100=0

      # ─── FP4 path selection ───
      - VLLM_USE_FLASHINFER_MOE_FP4=0
      - VLLM_TEST_FORCE_FP8_MARLIN=0
      - VLLM_USE_FLASHINFER_SAMPLER=1
      # XS-specific: explicitly select the patched CUTLASS NVFP4 GEMM backend.
      - VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass

      # TurboQuant - for `kv-cache-dtype` use `fp8` 
      # - VLLM_USE_TURBOQUANT=1
      # - TURBOQUANT_KV_BITS=4
    volumes:
      # XS multimodal body (modelopt format, ~21 GB) — replaces the regular NVFP4 body
      - ~/models/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm:/models/xs
      # DFlash drafter (~3.46 GB)
      - ~/models/qwen36-27b-dflash:/models/dflash-drafter
      - ~/models/mods/fix-qwen3.6-enhanced-chat-template/fixed_chat_template-v5.jinja:/app/fixed_chat_template-v5.jinja

    entrypoint: ["vllm"]
    command:
      - serve
      - /models/xs
      - --served-model-name
      - qwen/qwen3.6-27b
      - --host
      - 0.0.0.0
      - --port
      - "8000"
      - --tensor-parallel-size
      - "1"
      - --dtype
      - bfloat16
      - --quantization
      - compressed-tensors
      - --kv-cache-dtype
      - auto
      - --max-model-len
      - "196608"
      - --max-num-seqs
      - "6"
      - --max-num-batched-tokens
      - "32768"
      - --gpu-memory-utilization
      - "0.65" ### only for DFlash - use between 0.65 to 0.72
      # - "0.52" ### only for MTP
      - --enable-chunked-prefill
      - --enable-prefix-caching
      - --generation-config
      - vllm
      - --override-generation-config
      - '{"temperature": 0.7, "top_p": 0.8, "top_k": 40, "presence_penalty": 0.0, "repetition_penalty": 1.0}'
      - --load-format
      - fastsafetensors
      - --mamba-block-size
      - "256"
      - --trust-remote-code
      - --enable-auto-tool-choice
      - --tool-call-parser
      - qwen3_coder
      - --reasoning-parser
      - qwen3
      - --attention-backend
      - flashinfer
      - --limit-mm-per-prompt
      - '{"image": 4, "video": 2}'
      - --mm-encoder-tp-mode
      - data
      - --mm-processor-cache-type
      - shm
      - --mm-shm-cache-max-object-size-mb
      - "256"
      - --chat-template
      - /app/fixed_chat_template-v5.jinja
      - --speculative-config
      - '{"method":"dflash","model":"/models/dflash-drafter","num_speculative_tokens":10}'
      # - '{"method": "mtp", "num_speculative_tokens": 3}'
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
