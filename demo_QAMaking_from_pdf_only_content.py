import gradio as gr
from gradio_pdf import PDF
import fitz
from chat import ChatModel
import time
import base64
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from text2vec import SentenceModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 减去最大值，防止溢出
    return exp_x / exp_x.sum(axis=0)



strlabelsindustry ={"10":"行业中重大政策事件",
                    "9":"行业中重大政策事件",
                    "8":"行业中影响力政策事件",
                    "7":"行业中影响力政策事件",
                    "6": "行业中区域性大影响力政策事件",
                    "5": "行业中区域性大影响力政策事件",
                    "4": "行业中区域性中影响力政策事件",
                    "3": "行业中区域性中影响力政策事件",
                    "2": "行业中区域性小影响力政策事件",
                    "1": "行业中区域性小影响力政策事件",
                    "0": "行业中不重要事件",
                    }

strlabelscompany ={"10":"公司经营面或信用面重大事件",
                   "9":"公司经营面或信用面重大事件",
                   "8": "公司经营面或信用面影响力事件",
                   "7": "公司经营面或信用面影响力事件",
                   "6": "公司经营面或信用面大事件",
                   "5": "公司经营面或信用面大事件",
                   "4": "公司经营面或信用面中等事件",
                   "3": "公司经营面或信用面中等事件",
                   "2": "公司经营面或信用面小事件",
                   "1": "公司经营面或信用面小事件",
                   "0": "公司不重要事件",
                   }


def find_first_key_in_dicts(value):
    # 在 strlabelsindustry 中查找
    key = next((key for key, val in strlabelsindustry.items() if val == value), None)
    if key:
        return key

    # 在 strlabelscompany 中查找
    return next((key for key, val in strlabelscompany.items() if val == value), None)


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

image_base64 = encode_image_to_base64("icon/ATOM.png")


with open("/workspace/mnt/storage/xiangxin@supremind.com/infer_tensor_ceph/LM-playground_trt/template/QAtemplat.txt", 'r') as f:
    template = ''.join(f.readlines())





def _launch_demo(model, sentence_model, template):
    pdftxt = []

    def formatcontent(query: str) -> str:
        p = template + query
        return p

    def crawling_(content):
        p = formatcontent(content)

        message = {"role": "user", "content": p}

        allmessage = ""
        for new_text in model.stream_chat([message]):
            allmessage += new_text

        score = find_first_key_in_dicts(allmessage)

        #没有key的情况下
        if score == None:
            if "行业" in allmessage:
                embeddings = sentence_model.encode(allmessage)
                similarities = []
                for label, event in strlabelsindustry.items():
                    event_vector = sentence_model.encode(event)
                    similarity = cosine_similarity([embeddings], [event_vector])
                    # print(f"“{target_sentence}” 和 “{event}” 的相似度 (标签 {label}): {similarity[0][0]:.4f}")
                    similarities.append((label, similarity))
                
                # 取出相似度进行softmax计算
                similarity_values = [sim for _, sim in similarities]
                softmax_values = softmax(np.array(similarity_values))
                
                # 找到最大相似度和最大softmax值的标签
                max_similarity_label = similarities[np.argmax(similarity_values)][0]
                max_softmax_label = similarities[np.argmax(softmax_values)][0]

                if max_similarity_label == max_softmax_label:
                    score = max_similarity_label
                else:
                    score = -1
            else:
                embeddings = sentence_model.encode(allmessage)
                similarities = []
                for label, event in strlabelscompany.items():
                    event_vector = sentence_model.encode(event)
                    similarity = cosine_similarity([embeddings], [event_vector])
                    # print(f"“{target_sentence}” 和 “{event}” 的相似度 (标签 {label}): {similarity[0][0]:.4f}")
                    similarities.append((label, similarity))
                
                # 取出相似度进行softmax计算
                similarity_values = [sim for _, sim in similarities]
                softmax_values = softmax(np.array(similarity_values))
                
                # 找到最大相似度和最大softmax值的标签
                max_similarity_label = similarities[np.argmax(similarity_values)][0]
                max_softmax_label = similarities[np.argmax(softmax_values)][0]

                if max_similarity_label == max_softmax_label:
                    score = max_similarity_label
                else:
                    score = -1

        # print(allmessage)
            
        return allmessage + ":" + str(score)

    
    def loadpdf(path):
        #print(path)
        doc = fitz.open(path)
        for page in doc:
            pagetxt = []
            blocks = page.get_text('blocks')
            for block in blocks:
                if block[6] == 0:
                    pagetxt.append(block[4].replace("\n",""))
            
            pdftxt.append('\n'.join(pagetxt))
        # txt += text + "\n"
        print(len(pdftxt))
        #return pdftxt[0]
        return pdftxt[0]


    def  makeselection(content, evt: gr.SelectData):
        # print(content)
        # print(evt.value)
        return content + evt.value


    def add_text(history, task_history, query):
        if len(pdftxt) ==0:
            gr.Warning("请先上传pdf")
            return history, task_history, ""
        if len(query) == 0:
            gr.Warning("请从内容中框选用来出题的内容")
            return history, task_history, ""

        def formatcontent(prompt:str, query: str) -> str:
            qa_num = 3
            p = template.format(count=qa_num, text=query)
            return p
        if history is None:
            history = []
        history += [(formatcontent(template, query), None)]
        task_history += [task_history + [("111", None)]]
        #history = history + [(_parse_text(text), None)]
        #task_history = task_history + [(task_text, None)]
        #print(history)
        return history, task_history, ""


    def makeQA(_chatbot, task_history):
        if not _chatbot:
            return _chatbot
        #print(_chatbot)
        #import pdb
        #pdb.set_trace()
        chat_query = _chatbot[-1][0]
        query = task_history[-1][0]
        print("User: " + chat_query)

        message = {"role": "user", "content": chat_query}
        respones = ""
        start_time = time.time()
        for new_text in model.stream_chat([message]):
            respones += new_text
            #print(respones)
            _chatbot[-1] = (chat_query, respones)
            #print(_chatbot)
            #_chatbot += (p, new_text)
            yield _chatbot
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"代码运行时间: {execution_time} 秒")
        return _chatbot


    def reset_user_input():
        return gr.update(value="")


    def updateinfo(value):
        return pdftxt[value]


    def reset_state(task_history):
        task_history.clear()
        return []


    def logQA(value, bottom):
        import json 
        #print(bottom)
        savedict = {"question": value[-1][0], "answer": value[-1][1]}
        jsonstr = json.dumps(savedict, ensure_ascii=False)
        with open(bottom + ".json", "a", encoding="utf-8") as f:
            f.write(jsonstr + "\n")
        return
 

    def setbar():
        #bar.maximum = len(pdftxt)
        return gr.update(maximum=len(pdftxt)-1, value=0)


    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown(
                f"""
                <p align="center">
                <img src="data:image/png;base64,{image_base64}" alt="Logo" style="height: 80px;"/>
                </p>
                """
            )
            gr.Markdown("""<center><font size=8>ATOM</center>""")

        with gr.Row():
            gr_content = gr.Textbox(lines=2, label='内容')
        with gr.Row():
            result = gr.Textbox(label="结果", interactive=False)
        with gr.Row():
            run = gr.Button("Make Question")

        # chatbot = gr.Chatbot(label="QA")
        task_history = gr.State([])


        run.click(fn=crawling_, inputs = gr_content, outputs=[result])
    

    demo.launch(auth=("admin", "pass1234"), server_name="0.0.0.0" ,server_port=8111)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str, default="/workspace/mnt/storage/xiangxin@supremind.com/infer_tensor/qwen1.5-7B-chat/trt_engines/fp16/1-gpu/")
    parser.add_argument('--sentences_model_path', type=str, default="/workspace/mnt/storage/xiangxin@supremind.com/infer_tensor/qwen1.5-7B-chat/sentences_model")
    parser.add_argument('--tokenizer_dir', type=str, default="/workspace/mnt/storage/xiangxin@supremind.com/infer_tensor/qwen1.5-7B-chat")
    parser.add_argument('--prompt_path', type=str, default="/workspace/mnt/storage/xiangxin@supremind.com/infer_tensor/Qwen2-VL/smmc/LM-playground/template/QAtemplat.txt")
    parser.add_argument('--trt_llm', type=bool, default=True, help="whether use trt_llm" )

    args = parser.parse_args()

    #qwen trt
    args_chatmodel = {
        "model_name_or_path":args.model_name_or_path,
        "infer_backend":"trt-llm" if args.trt_llm else "huggingface",
        "tokenizer_dir":args.tokenizer_dir,
        "max_length":50,  
        "template": "default",  
        "trt_streaming": False,
        "temperature": 1,
        "top_p": 0.7,
        "top_k": 1,
        #"prompt_path": "/workspace/mnt/storage/xiangxin@supremind.com/infer_tensor_ceph/LM-playground_trt/template/QAtemplat.txt"
    }
    


    with open(args.prompt_path, 'r') as f:
        template = ''.join(f.readlines())

    model = ChatModel(args_chatmodel)
    sentence_model = SentenceModel(args.sentences_model_path)

    _launch_demo(model, sentence_model, template)

