import gradio as gr
from pathlib import Path
from gradio_image_annotation import image_annotator
import matplotlib.colors as mcolors
import tempfile


prompt_intention= "\
你是一个聪明的助手，旨在分析出文字中的作者的意图\n \
候选意图包括：\n\
1. 检测特定的目标\n\
2. 表达结果是否满意\n\
3. 开始训练\n\
4. 调整训练参数\n\
5. 其他的意图\n\
\n\
请从如下的文字中，选择一个候选意图作为输出，注意仅仅输出可能意图的序号即可\n\
----------------\n\
{text}\
"

prompt_detectobj = "请从如下的文字中分析出中，需要检测的目标是什么，返回的格式为:\n```\n检测：$YOUR_ANSWER_HERE\n```\n----------------\n{text}\n"

prompt_vldetection= "请完成如下任务:1. 判断图中是不是有{type}, 2.如果有框出来"

prompt_trainingflag = "你是一个聪明的助手，旨在分析出文字中的作者的意图:请从如下的文字中分析出中，是否满意当前结果，返回的格式为:\n```\n态度：$YOUR_ANSWER_HERE\n```\n----------------\n{text}\n"

oriimagepath = ""
renderimagepath = None


workingstatus = 0
# status 0: start 1: detectobj 2:detectionend 3: Training 4: Trainingend 

# status 0 - 1 need obj input 
obj = ""
# status 1 - 2, 3 need  flag to start training 
trainingflag = ""
# status 3 - 4 need annotation 
annotation = ""
# status 4 - 1 need lora file 
lorapath = Path(tempfile.gettempdir()) / "gradio"/"lora"
lorafile = ""
# vl predict flag
vl_flag= False

 
def pickcolor():
    import random 
    return random.choice([_ for _ in mcolors.TABLEAU_COLORS.keys()]) # init color


def formatcontent(prompt:str, query: str) -> str:
    qa_num = 3
    p = prompt.format(count=qa_num, text=query)
    return p


class vlmodel:
    def __init__(self):
        import os
        import tempfile
        self.lorafile = ""
        os.environ["CUDA_VISIBLE_DEVICES"] =  "0"
        self.model, self.tokenizer = self._loadvlmodel()
        self.uploaded_file_dir = os.environ.get("GRADIO_TEMP_DIR") or str(
        Path(tempfile.gettempdir()) / "gradio")
        self.PUNCTUATION = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."

    def _parse_text(self, text):
        lines = text.split("\n")
        lines = [line for line in lines if line != ""]
        count = 0
        for i, line in enumerate(lines):
            if "```" in line:
                count += 1
                items = line.split("`")
                if count % 2 == 1:
                    lines[i] = f'<pre><code class="language-{items[-1]}">'
                else:
                    lines[i] = f"<br></code></pre>"
            else:
                if i > 0:
                    if count % 2 == 1:
                        line = line.replace("`", r"\`")
                        line = line.replace("<", "&lt;")
                        line = line.replace(">", "&gt;")
                        line = line.replace(" ", "&nbsp;")
                        line = line.replace("*", "&ast;")
                        line = line.replace("_", "&lowbar;")
                        line = line.replace("-", "&#45;")
                        line = line.replace(".", "&#46;")
                        line = line.replace("!", "&#33;")
                        line = line.replace("(", "&#40;")
                        line = line.replace(")", "&#41;")
                        line = line.replace("$", "&#36;")
                    lines[i] = "<br>" + line
        text = "".join(lines)
        return text


    def _remove_image_special(self, text):
        import re
        text = text.replace('<ref>', '').replace('</ref>', '')
        return re.sub(r'<box>.*?(</box>|$)', '', text)


    def predict(self, _chatbot, task_history):
        import copy
        import secrets
        global renderimagepath
        global vl_flag
        if vl_flag == False:
            yield _chatbot
        else:
            if lorafile != "":
                if self.lorafile != lorafile:
                    if self.model is not None:
                        del self.model
                    self.model, self.tokenizer = self._loadvlmodel()
                    self.lorafile = lorafile
            try:
                chat_query = _chatbot[-1][0]
            except IndexError:
                import pdb
                pdb.set_trace()
            
            query = task_history[-1][0]
            print("User: " + self._parse_text(query))
            history_cp = copy.deepcopy(task_history)
            full_response = ""

            history_filter = []
            pic_idx = 1
            pre = ""
            #print(history_cp)
            #import pdb
            #pdb.set_trace()
            for i, (q, a) in enumerate(history_cp):
                if isinstance(q, (tuple, list)):
                    q = f'Picture {pic_idx}: <img>{q[0]}</img>'
                    pre += q + '\n'
                    pic_idx += 1
                else:
                    pre += q
                    history_filter.append((pre, a))
                    pre = ""
            history, message = history_filter[:-1], history_filter[-1][0]
            # response, history = model.chat(tokenizer, message, history=history)
            for response in self.model.chat_stream(self.tokenizer, message, history=history):
                _chatbot[-1] = (self._parse_text(chat_query), self._remove_image_special(self._parse_text(response)))
                yield _chatbot
                full_response = self._parse_text(response)

            response = full_response
            history.append((message, response))
            image = self.tokenizer.draw_bbox_on_latest_picture(response, history)

            if image is not None:
                temp_dir = secrets.token_hex(20)
                temp_dir = Path(self.uploaded_file_dir) / temp_dir
                temp_dir.mkdir(exist_ok=True, parents=True)
                name = f"tmp{secrets.token_hex(5)}.jpg"
                filename = temp_dir / name
                image.save(str(filename))
                #_chatbot.append((None, (str(filename),)))
                renderimagepath = filename
            else:
                _chatbot[-1] = (self._parse_text(chat_query), response)
                renderimagepath = None
            # full_response = _parse_text(response)

            task_history[-1] = (query, full_response)
            print("Qwen-VL-Chat: " + self._parse_text(full_response))
            yield _chatbot


    def _loadvlmodel(self):
        checkpoint_path = "/workspace/mnt/storage/zhaozhijian/yara/Qwen-VL/qwenchat/"
        global lorafile
        if lorafile == "":
            from modelscope import (
                snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
            )
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path, trust_remote_code=True, resume_download=True, revision='master',
            )

            device_map = "cuda"

            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                device_map=device_map,
                trust_remote_code=True,
                resume_download=True,
                revision='master',
            ).eval()
            model.generation_config = GenerationConfig.from_pretrained(
                checkpoint_path, trust_remote_code=True, resume_download=True, revision='master',
            )
        else:
            from peft import AutoPeftModelForCausalLM
            from modelscope import (
                AutoModelForCausalLM, AutoTokenizer, GenerationConfig
            )
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path, trust_remote_code=True, resume_download=True, revision='master',
            )
            model = AutoPeftModelForCausalLM.from_pretrained(
                lorafile, # path to the output directory
                device_map="cuda",
                trust_remote_code=True
            ).eval()
            model.generation_config = GenerationConfig.from_pretrained(
                checkpoint_path, trust_remote_code=True, resume_download=True, revision='master',
            )

        return model, tokenizer
    

    def fetch_all_box_with_ref(self, response):
        return self.tokenizer._fetch_all_box_with_ref(response)


class chatmodel:
    def __init__(self):
        self.model = self._loadmodel()


    def _loadmodel(self):
        global template
        from chat import ChatModel
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] =  "0"
        args = {}
        args['model_name_or_path'] = "/workspace/mnt/storage/zhaozhijian/yara/LLMModels/Qwen1___5-7B-Chat/"
        args['template'] = "qwen"
        args['prompt_path'] = "/workspace/mnt/storage/zhaozhijian/yara/LLaMA-Factory/template/QAtemplat.txt"
        with open(args['prompt_path'], 'r') as f:
            template = ''.join(f.readlines())

        model = ChatModel(args)
        return model
    
    def predict(self, _chatbot, task_history=None):
        if not _chatbot:
            return _chatbot
        #print(_chatbot)

        chat_query = _chatbot[-1][0]
        query = task_history[-1][0]
        print("User: " + chat_query)

        message = {"role": "user", "content": chat_query}
        respones = ""
        for new_text in self.model.stream_chat([message]):
            respones += new_text
            #print(respones)
            _chatbot[-1] = (chat_query, respones)
            #print(_chatbot)
            #_chatbot += (p, new_text)
            yield _chatbot
        return _chatbot


    def predictwithouthistory(self, chat_query): 
        print("User: " + chat_query)
        message = {"role": "user", "content": chat_query}
        respones = ""
        for new_text in self.model.stream_chat([message]):
            respones += new_text
            #print(respones)
        return respones


    def parseIntention(self, query):
        return self.predictwithouthistory(prompt_intention.format(text=query))
       

    def _parse_text(self, text):
        return text.replace("<|im_end|>", "").replace("<|im_start|>", "").replace("<|im_sep|>", "")


chat = chatmodel()

vl = vlmodel()


def get_boxes_json(annotations):
    global annotation
    if "boxes" in annotations:
        annotation = annotations["boxes"]
        return gr.update(value=annotations["boxes"])
    else:
        return gr.update(value="")


def deal_text(chatbox, task_history, text, image):
    global obj, trainingflag, annotation, workingstatus, vl_flag, renderimagepath
    intention = chat.parseIntention(text)
 
    print("Intention: " + intention)
    

    label_list= []
    label_colors  = []
    vl_flag = False

    if "1" in intention  and (workingstatus ==0 or workingstatus==1):
        tempobj = chat.predictwithouthistory(prompt_detectobj.format(text=text))
        if "检测：" in tempobj:
            obj = tempobj.replace("检测：", "")
        else:
            obj = ""
        print("Object: " + obj)

    if "2" in intention and workingstatus ==1:
        tempobj = chat.predictwithouthistory(prompt_trainingflag.format(text=text))
        print("Training Flag: " + obj)
        if "态度" in tempobj:
            trainingflag = tempobj.replace("态度：", "")
        else:
            trainingflag = ""
        print("trainingflag: " + trainingflag)

    
    if workingstatus ==0 and obj != "":
        text = prompt_vldetection.format(type=obj)
        task_text = text
        if len(text) >= 2 and text[-1] in vl.PUNCTUATION and \
            text[-2] not in vl.PUNCTUATION:
            task_text = text[:-1]
        chatbox = chatbox + [(vl._parse_text(text), None)]
        task_history = task_history + [(task_text, None)]
        workingstatus = 1
        vl_flag = True
       # chatbox = vl.predict(chatbox, task_history)
        return chatbox, task_history, gr.update(value=""), gr.update(label_list=label_list, label_colors=label_colors)
    elif workingstatus ==1 and "1" in intention:
        text = prompt_vldetection.format(type=obj)
        task_text = text
        if len(text) >= 2 and text[-1] in vl.PUNCTUATION and \
            text[-2] not in vl.PUNCTUATION:
            task_text = text[:-1]
        if chatbox == []:
            chatbox = chatbox + [(vl._parse_text(text), None)]
            task_history = task_history + [(task_text, None)]
        else:
            chatbox[-1] = (vl._parse_text(text), None)
            task_history[-1] = (task_text, None)
        workingstatus = 1
        vl_flag = True
    elif workingstatus ==1 and trainingflag == "满意":
        chatbox[-1][1] = "结果为：{}".format(trainingflag)
        text = "训推一体机DEMO 演示完毕， 点击clear 清除记录可以重试"
        chatbox = chatbox + [(vl._parse_text(text), None)]
        task_history = task_history + [(text, None)]
        workingstatus = 2
        vl_flag = False
    elif workingstatus ==1 and trainingflag == "不满意":
        chatbox[-1][1] = "结果为：{}".format(trainingflag)
        text = "对于不满意的{text}, 请完成标注".format(text=obj)
        chatbox = chatbox + [(vl._parse_text(text), None)]
        task_history = task_history + [(text, None)]
        workingstatus =3
        renderimagepath = None
        vl_flag = False
        label_list=[obj]
        label_colors = [mcolors.TABLEAU_COLORS[pickcolor()] for _ in range(len(label_list))]
    elif workingstatus ==3 and annotation!= "":
        text = "标注信息为：{text}, 点击Training 开始训练".format(text=annotation)
        chatbox = chatbox + [(vl._parse_text(text), None)]
        task_history = task_history + [(text, None)]
        workingstatus = 4
        vl_flag = False


    return chatbox, task_history, gr.update(value=""), gr.update(
        label_list=label_list, label_colors=label_colors)
        

def postprocess(chatbox, task_history, image):
    global obj, trainingflag, annotation, workingstatus, renderimagepath, vl_flag, lorafile
    if workingstatus ==0 and obj == "":
        text = "请先输入检测目标类型: 例如检查图中的猫"
        chatbox = chatbox + [(vl._parse_text(text), None)]
        task_history = task_history + [(text, None)]
    if workingstatus ==1 and trainingflag == "":
        text = "请选择是否满意检测结果"
        chatbox = chatbox + [(vl._parse_text(text), None)]
        task_history = task_history + [(text, None)]
    if workingstatus ==3 or workingstatus==4:
        logtext = True
    else:
        logtext = False
    if workingstatus ==4 and lorafile == "":
        text = "训练中，请稍等"
        chatbox = chatbox + [(vl._parse_text(text), None)]
        task_history = task_history + [(text, None)]
    elif workingstatus ==4 and lorafile != "":
        text = "训练完成, 结果保存在{path}".format(path=lorafile)
        chatbox = chatbox + [(vl._parse_text(text), None)]
        task_history = task_history + [(text, None)]
        workingstatus = 0
        obj == ""
        trainingflag = ""
        annotation = ""
        renderimagepath = None
        vl_flag = False
    return chatbox, task_history, gr.update(visible=logtext), 


def setinput(input,task_history):
    global oriimagepath
    print(input)
    oriimagepath = input
    value_ = {"image": input}
    hasorigin = False
    for info in task_history:
        if type(info[0]) == tuple:
            hasorigin = True
    
    if hasorigin == False:
        task_history = task_history + [((input,), None)]
    else:
        for i in range(len(task_history)):
            if type(task_history[i][0]) == tuple:
                task_history[i] = ((input,),None)

    return gr.update(value=value_), task_history


def printinput(input):
    print(input)
    return 


def renderimage():
    #print(image)
    global renderimagepath
    global oriimagepath

    if renderimagepath is not None:
        print(renderimagepath)
        value_ = {"image": renderimagepath}
        return gr.update(value=value_)
    return gr.update(value={"image": oriimagepath})
    

def changelora(loradd):
    global lorapath
    import os 
    if len(os.listdir(lorapath)) ==0:
        return gr.update(
            [], label="Animal", info="lora models", visible=False
        )
    else:
        return gr.update(choices= os.listdir(lorapath) + ["不用lora"], label="lora", info="lora model!", visible =True
        )

def printlora(loradd):
    import os
    global lorafile
    if loradd == "不用lora":
        lorafile = ""
    else:
        lorafile = os.path.join(lorapath, loradd)
    return  lorafile

def makeqwvltrainjson(annotation, oriimagepath):
    import cv2
    import tempfile 
    sampledir = Path(tempfile.gettempdir()) / "dataset"
    image = cv2.imread(oriimagepath)
    imheight , imwidth, _ = image.shape 
    convectionList = []
    labeldict = {}
    for labelinfo in annotation:
        key  = labelinfo['label']
        if labelinfo['label'] not in labeldict:
            labeldict[labelinfo['label']] = []
        xmin = labelinfo['xmin']
        ymin = labelinfo['ymin']
        xmax = labelinfo['xmax']
        ymax = labelinfo['ymax']
        normalloc = (xmin * 1000.0 / imwidth, ymin * 1000.0 / imheight,
                xmax * 1000.0/imwidth, ymax * 1000.0 / imheight)
        import numpy as np
        normalloc = np.array(normalloc).astype('int')
        labeldict[key].append(normalloc)

    keys = ""
    for key in labeldict.keys():
        keys += key

    info = {}

    info['from'] = 'user'
    info['value'] = "Picture 1: <img>" + oriimagepath+ "</img>\n图中包含了" + keys + "吗？"
    convectionList.append(info.copy())
    info['from'] = 'assistant'
    info['value'] = "是的图中包含了" + keys
    convectionList.append(info.copy())

    
    for key in labeldict:
        info['from'] = 'user'
        info['value'] = "框出图中的" + key
        convectionList.append(info.copy())
        info['from'] = 'assistant'
        info['value'] = "<ref>" + key + "/<ref>"
        for loc in labeldict[key]:
            tempstr = "<box>"
            tempstr += "({},{})({},{})".format(loc[0], loc[1], loc[2], loc[3])
            tempstr += "</box>"
            info['value'] +=tempstr
        convectionList.append(info.copy())
    
    id = "identity_" + str(1)
    sample = {
        "id": id,
        "conversations": convectionList
        }
    totalsample = []
    totalsample.append(sample)
    import json 
    import os
    sampledir.mkdir(exist_ok=True, parents=True)
    with open(os.path.join(sampledir,"sample.json"), "w") as f:
        json.dump(totalsample,f, ensure_ascii=False)
    return os.path.join(sampledir, "sample.json")

    
def train_lora():
    global annotation, oriimagepath, lorafile
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] =  "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["DIR"] = "/workspace/mnt/storage/zhaozhijian/yara/Qwen-VL/finetune"

    ## make dataset 
    samplejsonpath = makeqwvltrainjson(annotation, oriimagepath)

    DATA = samplejsonpath
    MODEL = "/workspace/mnt/storage/zhaozhijian/yara/Qwen-VL/qwenchat/"

    OUTDIR = os.path.join(lorapath, os.path.basename(oriimagepath))

    trainingscript= """python Qwen-VL/finetune.py \
        --model_name_or_path {model} \
        --data_path {data} \
        --bf16 True \
        --fix_vit True \
        --output_dir {output_dir} \
        --num_train_epochs 200 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 100 \
        --save_total_limit 10 \
        --learning_rate 1e-5 \
        --weight_decay 0.0 \
        --adam_beta2 0.95 \
        --warmup_ratio 0.01 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --report_to "none" \
        --model_max_length 2048 \
        --lazy_preprocess True \
        --gradient_checkpointing \
        --use_lora
    """.format(data= DATA, model=MODEL, output_dir= OUTDIR)

    print(trainingscript)
    import os 
    gn = os.popen(trainingscript)
    ## 改成定长队列
    lineall = []
    for line in gn:
        yield line
    lorafile = OUTDIR


with gr.Blocks() as demo:
    workingstatus = 0
    task_history = gr.State([])
    gr.Markdown("""# VLTrainingDemo\n
                1.  点击upload 上传图片\n
                2.  输入文本，点击submit\n""")
    with gr.Row():
        with gr.Column():
            bot = gr.Chatbot(label="VLTrainingBot")
            text_output = gr.Textbox(label="LOG", visible=False)
        with gr.Column():
            image = image_annotator(label="Image")
            button_get = gr.Button("Get bounding boxes")
    loradd = gr.Dropdown(
            [], label="lora", info="select a lora file"
        )
    input_box = gr.Textbox(show_label=False, placeholder="Enter text and press enter...")

    
    with gr.Row():
        submit = gr.Button("Submit")
        clear = gr.Button("Clear")
        file = gr.UploadButton("Upload")
        train = gr.Button("Training")

        
    submit.click(deal_text, inputs=[bot, task_history, input_box, image], outputs=[bot, task_history, input_box, image]).success(
            vl.predict, [bot, task_history], [bot]).success(
            renderimage, [], [image]).success(
                postprocess, [bot, task_history, image], [bot, task_history, text_output]
            )
    file.upload(setinput, [file, task_history], [image, task_history])

    loradd.focus(changelora, inputs=[loradd], outputs=[loradd])
    loradd.change(printlora, inputs=[loradd], outputs=[])


    button_get.click(get_boxes_json, [image], [input_box])

    train.click(train_lora, [], [text_output]).success(
        postprocess, [bot, task_history, text_output], [bot, task_history, text_output]
    )



if __name__ == "__main__":
    demo.launch(server_name= "0.0.0.0",inbrowser=True,server_port=8501)
