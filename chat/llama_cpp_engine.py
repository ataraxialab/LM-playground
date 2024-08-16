import asyncio
import concurrent.futures
import os
from threading import Thread
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import GenerationConfig, TextIteratorStreamer

from .base_engine import BaseEngine, Response
from chat.model.llama_cpp_lm import Llama
from llama_cpp.llama_tokenizer import LlamaHFTokenizer

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from trl import PreTrainedModelWrapper

    from ..data import Template
    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


class llamaCppEngine(BaseEngine):
    def __init__(self,
                 model_args: "ModelArguments" = None,
                 data_args: "DataArguments" = None,
                 finetuning_args: "FinetuningArguments" = None,
                 generating_args: "GeneratingArguments" = None
                 ):
        self.can_generate = True
        self.model_name = model_args.model_name_or_path
        self.generating_args = generating_args.to_dict()
        self.generating_args["n_gpu_layers"] = -1
        self.generating_args["chat_format"] = data_args.template
        self.generating_args["max_tokens"] = 1024
        self.llama_tokenizer = LlamaHFTokenizer.from_pretrained(model_args.tokenizer_dir)
        self.runner = Llama(
            model_path=self.model_name,
            n_gpu_layers=self.generating_args["n_gpu_layers"],
            chat_format=self.generating_args["chat_format"],
            tokenizer=self.llama_tokenizer
        )

    async def start(self) -> None:
        self._semaphore = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENT", 1)))

    @staticmethod
    def run_llama_cpp(**kwargs):
        runner = kwargs.get('runner')
        messages = kwargs.get('messages')
        top_p = kwargs.get('top_p')
        top_k = kwargs.get('top_k')
        temperature = kwargs.get('temperature')
        streamer = kwargs.get('streamer')
        max_tokens = kwargs.get('max_tokens')

        # input_ids在create_chat_completion转换,tokenizer也在runner里面了
        runner.create_chat_completion(
            messages=messages,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens,
            streamer=streamer
        )

        # print(x["choices"][0]["message"]["content"])

    @staticmethod
    def _process_args(messages, top_p, top_k, temperature, max_tokens):
        gen_kwargs = {
            "messages": messages,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        return gen_kwargs

    @staticmethod
    @torch.inference_mode()
    def _stream_chat(
            runner,
            tokenizer,
            # template,
            generating_args: Dict[str, Any],
            messages: Sequence[Dict[str, str]],
            system: Optional[str] = None,
            tools: Optional[str] = None,
            input_kwargs: Optional[Dict[str, Any]] = {},
    ) -> Callable[[], str]:

        generating_kwargs = llamaCppEngine._process_args(
            messages, generating_args["top_p"], generating_args["top_k"], generating_args["temperature"],
            generating_args["max_tokens"]
        )

        gen_kwargs = {
            "runner": runner,
            **generating_kwargs,
        }

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        gen_kwargs["streamer"] = streamer
        thread = Thread(target=llamaCppEngine.run_llama_cpp, kwargs=gen_kwargs, daemon=True)
        thread.start()

        def stream():
            try:
                return streamer.__next__()
            except StopIteration:
                raise StopAsyncIteration()

        return stream

    async def stream_chat(
            self,
            messages: Sequence[Dict[str, str]],
            system: Optional[str] = None,
            tools: Optional[str] = None,
            **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        if not self.can_generate:
            raise ValueError("The current model does not support `stream_chat`.")

        loop = asyncio.get_running_loop()
        input_args = (
            self.runner,
            self.llama_tokenizer,
            # self.template,
            self.generating_args,
            messages,
            system,
            tools,
            input_kwargs,
        )
        async with self._semaphore:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                stream = self._stream_chat(*input_args)
                while True:
                    try:
                        yield await loop.run_in_executor(pool, stream)
                    except StopAsyncIteration:
                        break

    @staticmethod
    @torch.inference_mode()
    def _chat(
            model: "PreTrainedModel",
            tokenizer: "PreTrainedTokenizer",
            template: "Template",
            generating_args: Dict[str, Any],
            messages: Sequence[Dict[str, str]],
            system: Optional[str] = None,
            tools: Optional[str] = None,
            input_kwargs: Optional[Dict[str, Any]] = {},
    ) -> List["Response"]:
        gen_kwargs, prompt_length = HuggingfaceEngine._process_args(
            model, tokenizer, template, generating_args, messages, system, tools, input_kwargs
        )
        generate_output = model.generate(**gen_kwargs)
        response_ids = generate_output[:, prompt_length:]
        response = tokenizer.batch_decode(response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        results = []
        for i in range(len(response)):
            eos_index = (response_ids[i] == tokenizer.eos_token_id).nonzero()
            response_length = (eos_index[0].item() + 1) if len(eos_index) else len(response_ids[i])
            results.append(
                Response(
                    response_text=response[i],
                    response_length=response_length,
                    prompt_length=prompt_length,
                    finish_reason="stop" if len(eos_index) else "length",
                )
            )

        return results

    @staticmethod
    @torch.inference_mode()
    def _get_scores(
            model: "PreTrainedModelWrapper",
            tokenizer: "PreTrainedTokenizer",
            batch_input: List[str],
            input_kwargs: Optional[Dict[str, Any]] = {},
    ) -> List[float]:
        max_length = input_kwargs.pop("max_length", None)
        device = getattr(model.pretrained_model, "device", "cuda")
        inputs = tokenizer(
            batch_input,
            padding=True,
            truncation=True,
            max_length=max_length or getattr(model.config, "max_position_embeddings", 1024),
            return_tensors="pt",
            add_special_tokens=True,
        ).to(device)

        input_ids: torch.Tensor = inputs["input_ids"]
        _, _, values = model(**inputs, output_hidden_states=True, return_dict=True)

        if getattr(model.config, "model_type", None) == "chatglm":
            values = torch.transpose(values, 0, 1)

        scores = []
        for i in range(input_ids.size(0)):
            end_indexes = (input_ids[i] != tokenizer.pad_token_id).nonzero()
            end_index = end_indexes[-1].item() if len(end_indexes) else 0
            scores.append(values[i, end_index].nan_to_num().item())

        return scores

    async def chat(
            self,
            messages: Sequence[Dict[str, str]],
            system: Optional[str] = None,
            tools: Optional[str] = None,
            **input_kwargs,
    ) -> List["Response"]:
        if not self.can_generate:
            raise ValueError("The current model does not support `chat`.")

        loop = asyncio.get_running_loop()
        input_args = (
            self.model,
            self.tokenizer,
            self.template,
            self.generating_args,
            messages,
            system,
            tools,
            input_kwargs,
        )
        async with self._semaphore:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return await loop.run_in_executor(pool, self._chat, *input_args)

    async def get_scores(
            self,
            batch_input: List[str],
            **input_kwargs,
    ) -> List[float]:
        if self.can_generate:
            raise ValueError("Cannot get scores using an auto-regressive model.")

        loop = asyncio.get_running_loop()
        input_args = (self.model, self.tokenizer, batch_input, input_kwargs)
        async with self._semaphore:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return await loop.run_in_executor(pool, self._get_scores, *input_args)
