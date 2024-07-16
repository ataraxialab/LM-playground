/opt/conda/bin/pip install gradio_pdf PyMuPDF accelerate==0.27.2 peft==0.10.0 trl==0.8.1 gradio==4.21.0 transformers==4.37.2
/opt/conda/bin/python LM-playground/demo_QAMaking_from_pdf.py --model_name_or_path Qwen1.5-7B-chat --prompt_path LM-playground/template/QAtemplat.txt
