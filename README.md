# LM-playground
repo contains mutil LLM &amp; VL demo

# QAMaking Demo

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2.  挂载模型和选择template


## 3 运行 

```bash
python demo_QAMaking_from_pdf.py --model_name_or_path $MODEL_PATH --template qwen  --prompt_path "template/QAtemplat.txt"
```

# Training Bot Demo

## 启动suprevision 的 训练api， 获取对应的ip

## 3 运行 

```bash
python demo_AutoTrain.py --model_name_or_path $MODEL_PATH --template qwen  --ip $IP

``` 

# VL Training Bot Demo

## clone qwen repo
```bash
https://github.com/QwenLM/Qwen-VL
``` 

## 修改参数

1. 修改 ·chatmodel· 中的模型路径和类型
2. 修改 ·vlmodel· 中的 checkpoint_path 的模型路径


## 3 运行 

```bash
python demo_VlAutoTraining.py 

``` 


# jiuyang api server

## 下载模型 到本地

`
git lfs install
git lfs clone https://hf-mirror.com/shibing624/text2vec-base-chinese


`

## convert model to trt engine 
修改 ·modelconvert/trt-llm/run.sh· 中的模型路径和类型
`
    python3 convert_checkpoint.py --model_dir ·../../../qwen2-7b-ppo200-new-1/· \
                                --output_dir ./tllm_checkpoint_1gpu_fp16 \
                                --dtype float16 

    trtllm-build --checkpoint_dir ./tllm_checkpoint_1gpu_fp16 \
                --output_dir ./qwen1.5-7B-chat/trt_engines/fp16/1-gpu \
                --max_input_len ·4096· \
                --gather_generation_logits  \
                --logits_dtype float16 \
                --gemm_plugin float16
`

## 修改参数
修改 ·demo_QAMaking_from_pdf_only_content.py· 中的模型路径和类型

·model_name_or_path· 为刚刚生成的trt engine路径
·sentences_model_path· 为刚刚下载的text2vec-base-chinese路径
·tokenizer_dir· 为刚刚下载的qwen2-7b-ppo200-new-1路径
·prompt_path· 为template/QAtemplat.txt路径


`
python3 demo_QAMaking_from_pdf_only_content.py --model_name_or_path modelconvert/trt-llm/qwen1.5-7B-chat/trt_engines/fp16/1-gpu/ --sentences_model_path text2vec-base-chinese/ --tokenizer_dir ../qwen2-7b-ppo200-new-1/ --prompt_path template/QAtemplat.txt

`


## 3 运行 

```bash
bash jiuyang_api.sh

``` 