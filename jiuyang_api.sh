#! /bin/bash
pip3 install -r requirements_jiuyangapi.txt -i https://pypi.mirrors.ustc.edu.cn/simple/

python3 demo_QAMaking_from_pdf_only_content.py --model_name_or_path modelconvert/trt-llm/qwen1.5-7B-chat/trt_engines/fp16/1-gpu/ --sentences_model_path text2vec-base-chinese/ --tokenizer_dir ../qwen2-7b-ppo200-new-1/ --prompt_path template/QAtemplat.txt

