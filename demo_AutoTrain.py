import gradio as gr
import requests
from chat import ChatModel

ngpu = 0
def _launch_demo(model, ip):
    with gr.Blocks() as demo:
        task_history = gr.State([("", "我是自动训练助手，我可以帮你完成suprevision的训练，训练需要指定GPU的数目，您可以从回答中添加GPU的数目，然后我们开始训练")])
        gr.Markdown("## auto training bot demo\n ### 使用方式\n 1. 通过对话确定训练参数（目前只支持GPU 数目）\n 2. 点击训练按钮开始训练\n 3. 点击停止按钮停止训练")
        with gr.Row():
            chatbot = gr.Chatbot(label="LLM训练机器人", value=task_history.value)
            text_output = gr.Textbox(label="LOG")
        with gr.Row():
            train = gr.Button("训练")
            stop = gr.Button("停止训练")
        
        chatinput = gr.Textbox(label="Input", value="如何选择使用几个GPU？")
        submit = gr.Button("Submit")

            
        def logbot():
            global ngpu
            if ngpu <1:
                return "GPU数目小于1"
            #ip = "10.207.7.13"
            port = "1235"
            url = "http://{ip}:{port}/training".format(ip=ip, port=port)
            print(url)
            with requests.get(url, stream=True, json={"lr": "0.01", "ngpu":ngpu}) as r:
                all = ""
                for chunk in r.iter_content(8192):  # or, for line in r.iter_lines():
                    all += chunk.decode('utf-8')
                    yield all


        def stopbot():
            #ip = "10.207.7.13"
            port = "1235"
            url = "http://{ip}:{port}/stoptraining".format(ip=ip, port=port)
            print(url)
            with requests.get(url) as r:
                print(r)
                return r.content


        def _chat(_chatinput, _chatbot, task_history):
            #print(task_history)
            if len(_chatinput)==0:
                return _chatbot
            #print(_chatbot)
            #import pdb
            #pdb.set_trace()
            messages = []
            for i, (q,a) in enumerate(task_history):
                message = {"role": "user", "content": q}
                messages.append(message)
                message = {"role": "assistant", "content": a}
                messages.append(message)
            #chat_query = _chatbot[-1][0]
            #query = task_history[-1][0]
            #print("User: " + chat_query)
            #print("history: " + query)
            print(_chatbot)
            _chatbot += [(_chatinput, "")]

            message = {"role": "user", "content": _chatinput}
            messages.append(message)
            respones = ""
            print("messages", messages)
            for new_text in model.stream_chat(messages):
                respones += new_text
                #print(respones)
                _chatbot[-1]= (_chatinput, respones)
                #print(_chatbot)
                #_chatbot += (p, new_text)
                yield "使用8个GPU", _chatbot
            task_history += [(_chatinput, respones)]
            #chatinput.value="使用8个GPU"
            return "使用8个GPU", _chatbot


        def checkparam(_chatbot, task_history):
            global ngpu
            prompt="""你是一个训练机器人，请从之前对话的历史中获取训练必须的参数，参数名称为{param}, 如果确定参数，请返回参数，格式为：
            ```
            {param}: 参数值1
            ```
            如果不确定，请返回“不确定”
            ----------------
            {text}
            """
            content = ""
            for i, (q,a) in enumerate(task_history):
                content +=q 
                content += a
            print(content)
            message = {"role": "user", "content": prompt.format(text=content, param="GPU数量")}
            respones = ""
            for new_text in model.stream_chat([message]):
                respones += new_text
                #print(respones)
            print(respones)
            import re
            match = re.search(r"GPU数量: (\d+)", respones)
            if match:
                ngpu = int(match.group(1))
                # add start training word
                success = "训练参数已经获取，{param} : {ngpu}， 点击Training，开始训练".format(param="GPU数量", ngpu=ngpu)
                ori_input , _ = task_history[-1] 
                task_history[-1] = (ori_input, success)
                _chatbot[-1] = (ori_input, success)
            return _chatbot


        train.click(logbot, inputs=[], outputs=[text_output])
        stop.click(stopbot, inputs=[], outputs=[text_output])
        submit.click(_chat, inputs=[chatinput,chatbot, task_history], 
                    outputs=[chatinput, chatbot]).success(checkparam, inputs=[chatbot, task_history], outputs=[chatbot])
        
        #submit.click(lambda x: gr.update(value=''), [],[inp])



    demo.launch(server_name="0.0.0.0", share=False, debug=True, server_port=8501)  # share=True to share the link



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="/workspace/mnt/storage/zhaozhijian/yara/LM-playground/Qwen1___5-7B/")
    parser.add_argument('--template', type=str, default="qwen")
    parser.add_argument('--ip', type=str, default="10.207.7.13")

    args = parser.parse_args()

    argsmodel = {}
    argsmodel["model_name_or_path"] = args.model_name_or_path
    argsmodel["template"] = args.template
    model = ChatModel(argsmodel)

    _launch_demo(model, args.ip)




#demo.launch(share=False, debug=True, server_port=8501)  # share=True to share the link