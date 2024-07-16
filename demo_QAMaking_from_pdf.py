import gradio as gr
from gradio_pdf import PDF
import fitz
from chat import ChatModel


def _launch_demo(model, template):
    pdftxt = []
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
        for new_text in model.stream_chat([message]):
            respones += new_text
            #print(respones)
            _chatbot[-1] = (chat_query, respones)
            #print(_chatbot)
            #_chatbot += (p, new_text)
            yield _chatbot
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
            pdf = PDF(label="Upload a PDF", interactive=True)
            with gr.Column():
                name=gr.Textbox(label="内容")
                bar = gr.Slider(label="Page", minimum=0, maximum=0, step=1,interactive=True)
                content = gr.Textbox("出题的内容")
        with gr.Row():
            run = gr.Button("Make Question")
            empty_bin = gr.Button("Clear")
            QA_good = gr.Button("题出的好")
            QA_bad = gr.Button("题出的不好")
        name.select(makeselection, content, content)
        pdf.upload(loadpdf, pdf, name).success(setbar, None, bar)  
        bar.change(updateinfo, inputs=bar, outputs=name, api_name="updateinfo")

        chatbot = gr.Chatbot(label="QA")
        task_history = gr.State([])
        run.click(add_text, [chatbot, task_history, content], [chatbot, task_history]).success(
                makeQA, [chatbot, task_history], [chatbot], show_progress=True
            )
        #run.click(reset_user_input, [], [content])
        #run.click(reset_user_input, [], [chatbot])
        empty_bin.click(reset_state, [task_history], [chatbot, content], show_progress=True)
        QA_good.click(logQA, [chatbot, QA_good], [])
        QA_bad.click(logQA, [chatbot,QA_bad], [])
        #run.click(makeQA, [chatbot, content], [chatbot], show_progress=True)

    demo.launch(server_name="0.0.0.0" ,server_port=8504)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str, default="/workspace/mnt/storage/zhaozhijian/pt-fs/Qwen1___5-7B-Chat/")
    parser.add_argument('--template', type=str, default="qwen")
    parser.add_argument('--prompt_path', type=str, default="template/QAtemplat.txt")

    args = parser.parse_args()

    with open(args.prompt_path, 'r') as f:
        template = ''.join(f.readlines())
    
    #print(vars(args).keys())

    model = ChatModel(vars(args))

    _launch_demo(model, template)


