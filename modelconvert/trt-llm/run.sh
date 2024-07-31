python3 convert_checkpoint.py --model_dir ../llmmodel/Qwen1.5-7B-Chat/ \
                              --output_dir ./tllm_checkpoint_1gpu_fp16 \
                              --dtype float16 

trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp16 \
            --output_dir ./qwen1.5-7B-chat/trt_engines/fp16/1-gpu \
            --max_output_len 1024 \
            --gather_generation_logits  \
            --logits_dtype float16 \
            --gemm_plugin float16