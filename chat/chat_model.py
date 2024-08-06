import asyncio
from threading import Thread
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Generator, List, Optional, Sequence

from .hparams import get_infer_args
from .hf_engine import HuggingfaceEngine
from .vllm_engine import VllmEngine
from chat import run
from chat.trt_llm_engine import trtLLMQwen
from .llama_cpp_engine import llamaCppQwen
if TYPE_CHECKING:
    from .base_engine import BaseEngine, Response


def _start_background_loop(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


class ChatModel:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        if args['trt_llm_llama'] == '':
            del args['trt_llm_llama']
            model_args, data_args, finetuning_args, generating_args = get_infer_args(args)
            model_args.infer_backend = "llama"
            if model_args.infer_backend == "huggingface":
                self.engine: "BaseEngine" = HuggingfaceEngine(model_args, data_args, finetuning_args, generating_args)
            elif model_args.infer_backend == "vllm":
                self.engine: "BaseEngine" = VllmEngine(model_args, data_args, finetuning_args, generating_args)
            elif model_args.infer_backend == "llama":
                self.engine: "BaseEngine" = llamaCppQwen(model_args, data_args, finetuning_args, generating_args)
            else:
                raise NotImplementedError("Unknown backend: {}".format(model_args.infer_backend))
        else:
            # 不用更新
            trt_llm_args = run.parse_arguments(args_copy=args)

            args_copy = SimpleNamespace(**args)

            args, is_enc_dec, runtime_rank, tokenizer, end_id, pad_id, stop_words_list, bad_words_list, model_name, model_version, prompt_template = run.get_trt_llm_base_args(
                trt_llm_args)

            self.engine: "BaseEngine" = trtLLMQwen(args_copy=args_copy, args=args,
                                                   trt_llm_args=trt_llm_args,
                                                   is_enc_dec=is_enc_dec, runtime_rank=runtime_rank,
                                                   tokenizer=tokenizer, end_id=end_id, pad_id=pad_id,
                                                   stop_words_list=stop_words_list,
                                                   bad_words_list=bad_words_list, model_name=model_name,
                                                   model_version=model_version, prompt_template=prompt_template
                                                   )

        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=_start_background_loop, args=(self._loop,), daemon=True)
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self.engine.start(), self._loop)

    def chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        **input_kwargs,
    ) -> List["Response"]:
        task = asyncio.run_coroutine_threadsafe(self.achat(messages, system, tools, **input_kwargs), self._loop)
        return task.result()

    async def achat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        **input_kwargs,
    ) -> List["Response"]:
        return await self.engine.chat(messages, system, tools, **input_kwargs)

    def stream_chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        **input_kwargs,
    ) -> Generator[str, None, None]:
        generator = self.astream_chat(messages, system, tools, **input_kwargs)
        while True:
            try:
                task = asyncio.run_coroutine_threadsafe(generator.__anext__(), self._loop)
                yield task.result()
            except StopAsyncIteration:
                break

    async def astream_chat(
        self,
        messages: Sequence[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        **input_kwargs,
    ) -> AsyncGenerator[str, None]:
        async for new_token in self.engine.stream_chat(messages, system, tools, **input_kwargs):
            yield new_token

    def get_scores(
        self,
        batch_input: List[str],
        **input_kwargs,
    ) -> List[float]:
        task = asyncio.run_coroutine_threadsafe(self.aget_scores(batch_input, **input_kwargs), self._loop)
        return task.result()

    async def aget_scores(
        self,
        batch_input: List[str],
        **input_kwargs,
    ) -> List[float]:
        return await self.engine.get_scores(batch_input, **input_kwargs)
