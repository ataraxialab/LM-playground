## trt-llm 转模型

### docker 镜像

reg.supremind.info/algorithmteam/suprellm:trtllmv3

### 使用方法

#### engine 转换
--model_dir 为原始的权重保存的路径
 --output_dir 为safetensors 的保存路径


```python
python3 convert_checkpoint.py --model_dir ../llmmodel/Qwen1.5-7B-Chat/ \
                              --output_dir ./tllm_checkpoint_1gpu_fp16 \
                              --dtype float16 
```

--checkpoint_dir 为上个命令保存的safetensors 路径
--output_dir 为trt engine 保存的路径
--max_output_len 为生成的最大长度
--gather_generation_logits 为是否gather logits
--logits_dtype 为logits 的数据类型
--gemm_plugin 为是否使用gemm plugin

```
trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp16 \
            --output_dir ./qwen1.5-7B-chat/trt_engines/fp16/1-gpu \
            --max_output_len 1024 \
            --gather_generation_logits  \
            --logits_dtype float16 \
            --gemm_plugin float16
```

#### engnie 测试

--input_text 为输入的文本
--max_output_len 为生成的最大长度
--tokenizer_dir 为tokenizer 的路径
--kv_cache_free_gpu_memory_fraction 为kv cache 占用显存的比例
--engine_dir 为trt engine 的路径

···python
python3 run.py --input_text "你好，请问你叫什么？" \
                  --max_output_len 50 \
                  --tokenizer_dir ../llmmodel/Qwen1.5-7B-Chat/ \
                  --kv_cache_free_gpu_memory_fraction 0.9 \
                  --engine_dir ./qwen1.5-7B-chat/trt_engines/fp16/1-gpu/
···