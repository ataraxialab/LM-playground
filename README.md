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
