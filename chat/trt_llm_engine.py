from tensorrt_llm.runtime import ModelRunnerCpp
from .base_engine import BaseEngine
from tensorrt_llm.builder import get_engine_version
import csv
import numpy as np
from pathlib import Path

import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS

import asyncio
import concurrent.futures
import os
from threading import Thread
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, List, Optional, Sequence

import torch
from transformers import GenerationConfig, TextIteratorStreamer
from .model import load_model, load_tokenizer, load_tokenizer_trt
from .base_engine import BaseEngine, Response

INTERNLM_META_INSTRUCTION = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

DEFAULT_PROMPT_TEMPLATES = {
    'InternLMForCausalLM':
    "<|User|>:{input_text}<eoh>\n<|Bot|>:",
    'InternLM2ForCausalLM':
    "<|im_start|>system\n" + INTERNLM_META_INSTRUCTION +
    "<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n",
    'QWenForCausalLM':
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n",
}


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from trl import PreTrainedModelWrapper

    from ..data import Template
    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def read_model_name(engine_dir: str):
    engine_version = get_engine_version(engine_dir)

    with open(Path(engine_dir) / "config.json", 'r') as f:
        import json
        config = json.load(f)

    if engine_version is None:
        return config['builder_config']['name'], None

    model_arch = config['pretrained_config']['architecture']
    model_version = None
    if model_arch == 'ChatGLMForCausalLM':
        model_version = config['pretrained_config']['chatglm_version']
    if model_arch == 'QWenForCausalLM':
        model_version = config['pretrained_config']['qwen_type']
    return model_arch, model_version

def supports_inflight_batching(engine_dir):
    from tensorrt_llm.bindings import GptJsonConfig
    config_path = Path(engine_dir) / "config.json"
    json_config = GptJsonConfig.parse_file(config_path)
    model_config = json_config.model_config
    return model_config.supports_inflight_batching

def parse_input(tokenizer,
                input_text=None,
                prompt_template=None,
                input_file=None,
                add_special_tokens=True,
                max_input_length=923,
                pad_id=None,
                num_prepend_vtokens=[],
                model_name=None,
                model_version=None):
    if pad_id is None:
        pad_id = tokenizer.pad_token_id

    batch_input_ids = []
    if input_file is None:
        for curr_text in input_text:
            if prompt_template is not None:
                curr_text = prompt_template.format(input_text=curr_text)
            input_ids = tokenizer.encode(curr_text,
                                         add_special_tokens=add_special_tokens,
                                         truncation=True,
                                         max_length=max_input_length)
            batch_input_ids.append(input_ids)
    else:
        if input_file.endswith('.csv'):
            with open(input_file, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    input_ids = np.array(line, dtype='int32')
                    batch_input_ids.append(input_ids[-max_input_length:])
        elif input_file.endswith('.npy'):
            inputs = np.load(input_file)
            for row in inputs:
                input_ids = row[row != pad_id]
                batch_input_ids.append(input_ids[-max_input_length:])
        elif input_file.endswith('.txt'):
            with open(input_file, 'r', encoding='utf-8',
                      errors='replace') as txt_file:
                input_text = txt_file.readlines()
                batch_input_ids = tokenizer(
                    input_text,
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    max_length=max_input_length)["input_ids"]
        else:
            print('Input file format not supported.')
            raise SystemExit

    if num_prepend_vtokens:
        assert len(num_prepend_vtokens) == len(batch_input_ids)
        base_vocab_size = tokenizer.vocab_size - len(
            tokenizer.special_tokens_map.get('additional_special_tokens', []))
        for i, length in enumerate(num_prepend_vtokens):
            batch_input_ids[i] = list(
                range(base_vocab_size,
                      base_vocab_size + length)) + batch_input_ids[i]

    if model_name == 'ChatGLMForCausalLM' and model_version == 'glm':
        for ids in batch_input_ids:
            ids.append(tokenizer.sop_token_id)

    batch_input_ids = [
        torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
    ]
    return batch_input_ids

def print_output(tokenizer,
                 output_ids,
                 input_lengths,
                 sequence_lengths,
                 output_csv=None,
                 output_npy=None,
                 context_logits=None,
                 generation_logits=None,
                 cum_log_probs=None,
                 log_probs=None,
                 output_logits_npy=None,
                 output_cum_log_probs_npy=None,
                 output_log_probs_npy=None, streamer = None):
        batch_size, num_beams, _ = output_ids.size()
        for batch_idx in range(batch_size):
            inputs = output_ids[batch_idx][0][:input_lengths[batch_idx]].tolist(
            )
            input_text = tokenizer.decode(inputs)
            print(f'Input [Text {batch_idx}]: \"{input_text}\"')
            for beam in range(num_beams):
                output_begin = input_lengths[batch_idx]
                output_end = sequence_lengths[batch_idx][beam]

                # trt-llm的
                # outputs = output_ids[batch_idx][beam][
                #           output_begin:output_end].tolist()

                # output_text = tokenizer.decode(outputs)
                # print(
                #     f'Output [Text {batch_idx} Beam {beam}]: \"{output_text}\"')

                outputs1 = output_ids[batch_idx][beam][
                          output_begin:output_end]

                # if streamer is not None:
                # for tensor in outputs1.tolist():
                    # tensor = torch.tensor(tensor)
                if streamer is not None:
                    for item in outputs1:
                        streamer.put(torch.tensor([item]))

class TrtLLMEngine(BaseEngine):
    # def __init__(self, args_copy, args, trt_llm_args, is_enc_dec, runtime_rank, tokenizer, end_id, pad_id,
    #              stop_words_list, bad_words_list, model_name, model_version, prompt_template,
    #              model_args: "ModelArguments" = None,
    #              data_args: "DataArguments" = None,
    #              finetuning_args: "FinetuningArguments" = None,
    #              generating_args: "GeneratingArguments" = None
    #              ):
    def __init__(self,
                 model_args: "ModelArguments" = None,
                 data_args: "DataArguments" = None,
                 finetuning_args: "FinetuningArguments" = None,
                 generating_args: "GeneratingArguments" = None
                 ):
        
        self.is_enc_dec = {
            name
            for name in os.listdir(model_args.model_name_or_path)
            if os.path.isdir(os.path.join(model_args.model_name_or_path))
        } == {'encoder', 'decoder'}

        model_name, model_version = read_model_name(model_args.model_name_or_path
                                                    ) if not self.is_enc_dec else ("", "")

        if model_args.tokenizer_dir is None :
            logger.ValueError(
                "tokenizer_dir is not specified. Try to infer from model_name, but this may be incorrect."
            )

        self.tokenizer, self.pad_id, self.end_id = load_tokenizer_trt(
            tokenizer_dir=model_args.tokenizer_dir,
            vocab_file=model_args.vocab_file,
            model_name=model_name,
            model_version=model_version,
            tokenizer_type=model_args.tokenizer_type,
        )
        generating_args.pad_id = self.pad_id
        generating_args.end_id = self.end_id

        self.debug_mode = generating_args.trt_debug_mode
        self.return_all_generated_tokens = generating_args.trt_return_all_generated_tokens

        self.stop_words_list = None
        generating_args.stop_words_list = self.stop_words_list
        # if args.stop_words:
        #     stop_words_list = tensorrt_llm.runtime.decode_words_list(
        #         args.stop_words, tokenizer)

        self.bad_words_list = None
        # if args.bad_words:
        #     bad_words_list = tensorrt_llm.runtime.decode_words_list(
        #         args.bad_words, tokenizer)
        generating_args.bad_words_list = self.bad_words_list

        self.prompt_template = None
        if  model_name in DEFAULT_PROMPT_TEMPLATES:
            self.prompt_template = DEFAULT_PROMPT_TEMPLATES[model_name]

        config  = {}
        engine_dir = Path(model_args.model_name_or_path)
        config_path = os.path.join(engine_dir, "config.json")
        with open(config_path, 'r') as f:
            import json
            config = json.load(f)



        self.runner_kwargs = {
            'engine_dir': model_args.model_name_or_path,
            'lora_dir': None,
            'rank': 0,
            'debug_mode': False,
            'lora_ckpt_source': 'hf',
            'gpu_weights_percent': 1,
            'is_enc_dec': False,
            'max_batch_size': config['build_config']["max_batch_size"],
            'max_input_len': config['build_config']["max_input_len"],
            'max_output_len': generating_args.max_length,
            'max_beam_width': config['build_config']["max_beam_width"],
            'max_attention_window_size': generating_args.trt_max_attention_window_size,
            'sink_token_length': generating_args.trt_sink_token_length,
            'max_tokens_in_paged_kv_cache': generating_args.trt_max_tokens_in_paged_kv_cache,
            'kv_cache_enable_block_reuse': generating_args.trt_kv_cache_enable_block_reuse,
            'kv_cache_free_gpu_memory_fraction': generating_args.trt_kv_cache_free_gpu_memory_fraction,
            'enable_chunked_context': generating_args.trt_enable_chunked_context,
        }
        
        #self.generating_args = generating_args
        self.runtime_rank = tensorrt_llm.mpi_rank()

        self.model_name = model_name
        self.model_version = model_version
        self.can_generate = True
        self.tokenizer.padding_side = "left"
        #self.template = get_template_and_fix_tokenizer(self.tokenizer, 'qwen')
        self.runner = ModelRunnerCpp.from_dir(**self.runner_kwargs)

        ### init the running para
        self.trt_return_all_generated_tokens = generating_args.trt_return_all_generated_tokens
        self.support_inflight_batching = supports_inflight_batching(
            os.path.join(model_args.model_name_or_path, "decoder") if self.is_enc_dec else model_args.
                model_name_or_path
                )
        
        if not self.support_inflight_batching:
            logger.warning(
                "The given engine does not support in-flight batching, fallback to python session"
            )
            self.use_py_session = True

        if not PYTHON_BINDINGS and not self.use_py_session:
            logger.warning(
                "Python bindings of C++ session is unavailable, fallback to Python session."
            )
            self.use_py_session = True
        if self.debug_mode and not self.use_py_session:
            logger.warning(
                "Debug mode is not supported in C++ session for now, fallback to Python session."
            )
            self.use_py_session = True
        
        if self.trt_return_all_generated_tokens and self.use_py_session:
            raise ValueError(
                "Returning all the generated tokens at each step is not supported in the Python session, use C++ session instead."
            )
        if (not self.return_all_generated_tokens) and generating_args.trt_streaming and (
                generating_args.trt_num_beams > 1):
            logger.warning(
                "Setting return_all_generated_tokens to True since streaming AND beam search are done simultaneously. "
                "Returning the full beams at each streaming step is needed because beam search + streaming can change previous outputs. "
                "WARNING: using this option may increase network usage significantly (quadratically w.r.t output length)."
            )
            self.return_all_generated_tokens = True

        
        
        self.generating_args = generating_args


    @staticmethod
    def run_trt_llm(**kwargs):
        runner_trt_llm = kwargs.get('runner')
        #args = kwargs.get('args')
        tokenizer = kwargs.get('tokenizer')
        batch_input_ids = kwargs.get('batch_input_ids')

        input_lengths = kwargs.get('input_lengths')

        streamer =  kwargs.get('streamer') if 'streamer' in kwargs else None

        # 把batch input ids去掉
        if streamer is not None:
            streamer.put(batch_input_ids[0].cpu())

        # 相差结果大不大，
        with torch.no_grad():
            outputs = runner_trt_llm.generate(**kwargs)
            torch.cuda.synchronize()

        # if runtime_rank == 0:这里为什么一开始就输入一次input-ids
        output_ids = outputs['output_ids']
        # if streamer is not None:
        #     streamer.put(output_ids[0,:,:].cpu())

        sequence_lengths = outputs['sequence_lengths']
        context_logits = None
        generation_logits = None
        cum_log_probs = None
        log_probs = None
        if runner_trt_llm.gather_context_logits:
            context_logits = outputs['context_logits']
        if runner_trt_llm.gather_generation_logits:
            generation_logits = outputs['generation_logits']
        if kwargs.get("output_cum_log_probs_npy") != None:
            cum_log_probs = outputs['cum_log_probs']
        if kwargs.get("output_log_probs_npy") != None:
            log_probs = outputs['log_probs']

        print_output(tokenizer,
                     output_ids,
                     input_lengths,
                     sequence_lengths,
                     output_csv=kwargs.get("output_csv") if "output_csv" in kwargs else None,
                     output_npy=kwargs.get("output_npy") if "output_npy" in kwargs else None,
                     context_logits=context_logits,
                     generation_logits=generation_logits,
                     output_logits_npy=kwargs.get("output_logits_npy") if "output_logits_npy" in kwargs else None,
                     cum_log_probs=cum_log_probs,
                     log_probs=log_probs,
                     output_cum_log_probs_npy=kwargs.get("output_cum_log_probs_npy") if "output_cum_log_probs_npy" in kwargs else None,
                     output_log_probs_npy=kwargs.get("output_log_probs_npy") if "output_log_probs_npy" in kwargs else None,
                     streamer = streamer)

        if streamer is not None:
            streamer.end()


    @staticmethod
    def _process_args(tokenizer, 
                      prompt_template, 
                      model_name, 
                      model_version, 
                      pad_id, 
                      genaration_kwargs,
                      num_prepend_vtokens=[],
                      max_input_length:int=923,
                      add_special_tokens:bool = True,
                      messages:Sequence[Dict[str, str]] = None):
        
        input_text = ""
        for mess in messages:
            input_text += mess['content'].strip()


        batch_input_ids = parse_input(tokenizer=tokenizer,
                                    input_text=[input_text],
                                    prompt_template=prompt_template,
                                    input_file=None,
                                    add_special_tokens=add_special_tokens,
                                    max_input_length=max_input_length,
                                    pad_id=pad_id,
                                    num_prepend_vtokens=num_prepend_vtokens,
                                    model_name=model_name,
                                    model_version=model_version)   


        input_lengths =  [x.size(0) for x in batch_input_ids]

        gen_kwargs ={
            "tokenizer":tokenizer,
            'batch_input_ids': batch_input_ids,
            "input_lengths": input_lengths,
            "encoder_input_ids": None,
            "max_new_tokens": max_input_length,
            "max_attention_window_size": genaration_kwargs.trt_max_attention_window_size,
            "sink_token_length": genaration_kwargs.trt_sink_token_length,
            "end_id": genaration_kwargs.end_id,
            "pad_id": genaration_kwargs.pad_id,
            "temperature": genaration_kwargs.temperature,
            "top_k": genaration_kwargs.top_k,
            "top_p": genaration_kwargs.top_p,
            "num_beams": genaration_kwargs.trt_num_beams,
            "length_penalty": genaration_kwargs.length_penalty,
            "early_stopping": genaration_kwargs.early_stopping,
            "repetition_penalty": genaration_kwargs.repetition_penalty,
            "presence_penalty": genaration_kwargs.trt_presence_penalty,
            "frequency_penalty": genaration_kwargs.trt_frequency_penalty,
            "stop_words_list": genaration_kwargs.stop_words_list,
            "bad_words_list": genaration_kwargs.bad_words_list,
            "output_cum_log_probs": genaration_kwargs.trt_output_cum_log_probs_npy!= None,
            "output_log_probs": genaration_kwargs.trt_output_log_probs_npy != None,
            "random_seed": genaration_kwargs.random_seed,
            "lora_uids": genaration_kwargs.trt_lora_task_uids,
            "prompt_table": genaration_kwargs.trt_prompt_table,
            "prompt_tasks": genaration_kwargs.trt_prompt_tasks,
            "streaming": genaration_kwargs.trt_streaming,
            "no_repeat_ngram_size":genaration_kwargs.trt_no_repeat_ngram_size,
            "medusa_choices":genaration_kwargs.trt_medusa_choices,
            "return_all_generated_tokens":genaration_kwargs.trt_return_all_generated_tokens,
            "output_sequence_lengths": True,
            "return_dict":True,
        }

        return gen_kwargs
    

        # runner_cls = ModelRunner if args.use_py_session else ModelRunnerCpp

        # if not args.use_py_session:
        #     runner_kwargs.update(is_enc_dec=is_enc_dec)
        # if args.medusa_choices is not None:
        #     args.medusa_choices = ast.literal_eval(args.medusa_choices)
        #     assert args.temperature == 1.0, "Medusa should use temperature == 1.0"
        #     assert args.num_beams == 1, "Medusa should use num_beams == 1"
        #     runner_kwargs.update(medusa_choices=args.medusa_choices)

        # return args, is_enc_dec, runtime_rank, tokenizer, batch_input_ids, encoder_input_ids, decoder_input_ids, input_lengths, \
        #     end_id, pad_id, stop_words_list, bad_words_list


    @staticmethod
    @torch.inference_mode()
    def _stream_chat(
            runner,
            tokenizer,
            prompt_template,
            model_name, 
            model_version, 
            pad_id, 
            messages: Sequence[Dict[str, str]],
            generating_args: Dict[str, Any],
            system: Optional[str] = None,
            tools: Optional[str] = None,
            input_kwargs: Optional[Dict[str, Any]] = {},
    ) -> Callable[[], str]:
        generating_kwargs = TrtLLMEngine._process_args(tokenizer,
                                                    prompt_template,
                                                    model_name,
                                                    model_version,
                                                    pad_id,
                                                    generating_args,
                                                    messages = messages)


        gen_kwargs = {
            "runner": runner,
            **generating_kwargs,
        }
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer
        thread = Thread(target=TrtLLMEngine.run_trt_llm, kwargs=gen_kwargs, daemon=True)
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
            self.tokenizer,
            self.prompt_template,
            self.model_name,
            self.model_version,
            self.pad_id,
            messages,
            self.generating_args,
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

    async def start(self) -> None:
        self._semaphore = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENT", 1)))

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
        gen_kwargs, prompt_length = TrtLLMEngine._process_args(
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
