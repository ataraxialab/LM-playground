python3 run.py --input_text "你好，请问你叫什么？" \
                  --max_output_len 50 \
                  --tokenizer_dir ../llmmodel/Qwen1.5-7B-Chat/ \
                  --kv_cache_free_gpu_memory_fraction 0.9 \
                  --engine_dir ./qwen1.5-7B-chat/trt_engines/fp16/1-gpu/