from chat import ChatModel

args = {
    "model_name_or_path":"modelconvert/qwen1.5-7B-chat/trt_engines/fp16/1-gpu/",
    "infer_backend":"trt-llm",
    "tokenizer_dir":"llmmodel/Qwen1.5-7B-Chat/",
    "max_length":50,  
    "template":"qwen",  
    "trt_streaming": False,
    "temperature": 1,
    "top_p": 0.0,
    "top_k": 1,
}
model = ChatModel(args)
message = {"role": "user", "content": "你好，请问你叫什么？"}
allmessage = ""
for new_text in model.stream_chat([message]):
    allmessage += new_text
print(allmessage)

argsqwen = {
    "model_name_or_path":"llmmodel/Qwen1.5-7B-Chat/",
    "template": "qwen"
}
model = ChatModel(argsqwen)
allmessage = ""
for new_text in model.stream_chat([message]):
    allmessage += new_text
print(allmessage)