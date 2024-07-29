from tensorrt_llm.runtime import ModelRunnerCpp
from .base_engine import BaseEngine
import argparse
import torch
from chat.utils import (DEFAULT_HF_MODEL_DIRS, DEFAULT_PROMPT_TEMPLATES,
                        add_common_args, load_tokenizer, read_decoder_start_token_id,
                        read_model_name, supports_inflight_batching,
                        throttle_generator)
from tensorrt_llm.runtime import (
    ModelConfig, SamplingConfig, GenerationSession
)
import ast
import csv
import numpy as np
from pathlib import Path

import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

import asyncio
import concurrent.futures
import os
from threading import Thread
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import GenerationConfig, TextIteratorStreamer

from .data import get_template_and_fix_tokenizer
from .extras.misc import get_logits_processor
from .model import load_model, load_tokenizer
from .base_engine import BaseEngine, Response

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from trl import PreTrainedModelWrapper

    from ..data import Template
    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


def add_common_args(parser):
    # sampling arguments
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams > 1",
                        default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--beam_search_diversity_rate', type=float, default=0.0)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--early_stopping',
                        type=int,
                        help='Use early stopping if num_beams > 1'
                             '1 for early-stopping, 0 for non-early-stopping'
                             'other values for stopping by length',
                        default=1)
    parser.add_argument(
        '--stop_words',
        default=None,
        type=str,
        nargs="+",
        action='append',
        help=
        'Set stop words for a batch. Successive invocations of --stop_words set stop words for other batches.'
        '    E.g.: --stop_words " London" " chef" --stop_words "eventually became" "was not"',
    )
    parser.add_argument(
        '--bad_words',
        default=None,
        type=str,
        nargs="+",
        action='append',
        help=
        'Set bad words for a batch. Successive invocations of --bad_words set bad words for other batches.'
        '    E.g.: --bad_words " London" " chef" --bad_words "eventually became" "was not"',
    )
    parser.add_argument('--no_repeat_ngram_size', type=int, default=None)

    # common runtime arguments
    parser.add_argument('--sink_token_length',
                        type=int,
                        default=None,
                        help='The sink token length.')
    parser.add_argument(
        '--max_attention_window_size',
        type=int,
        default=None,
        help=
        'The attention window size that controls the sliding window attention / cyclic kv cache behavior'
    )
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument(
        '--no_prompt_template',
        dest='use_prompt_template',
        default=True,
        action='store_false',
        help=
        "Whether or not to use default prompt template to wrap the input text.")
    parser.add_argument('--use_py_session',
                        default=False,
                        action='store_true',
                        help="Whether or not to use Python runtime session")
    parser.add_argument('--debug_mode',
                        default=False,
                        action='store_true',
                        help="Whether or not to turn on the debug mode")
    parser.add_argument('--streaming', default=False, action='store_true')
    parser.add_argument('--streaming_interval',
                        type=int,
                        help="How often to return tokens when streaming.",
                        default=5)
    parser.add_argument(
        '--prompt_table_path',
        type=str,
        help="Path to .npy file, exported by nemo_prompt_convert.py")
    parser.add_argument(
        '--prompt_tasks',
        help="Comma-separated list of tasks for prompt tuning, e.g., 0,3,1,0")
    parser.add_argument('--lora_dir',
                        type=str,
                        default=None,
                        nargs="+",
                        help="The directory of LoRA weights")
    parser.add_argument('--lora_ckpt_source',
                        type=str,
                        default="hf",
                        choices=["hf", "nemo"],
                        help="The source of lora checkpoint.")
    parser.add_argument(
        '--lora_task_uids',
        type=str,
        default=None,
        nargs="+",
        help="The list of LoRA task uids; use -1 to disable the LoRA module")
    parser.add_argument(
        '--num_prepend_vtokens',
        nargs="+",
        type=int,
        help="Number of (default) virtual tokens to prepend to each sentence."
             " For example, '--num_prepend_vtokens=10' will prepend the tokens"
             " [vocab_size, vocab_size + 1, ..., vocab_size + 9] to the sentence.")
    parser.add_argument(
        '--medusa_choices',
        type=str,
        default=None,
        help="Medusa choice to use, if not none, will use Medusa decoding."
             "   E.g.: [[0, 0, 0, 0], [0, 1, 0], [1, 0], [1, 1]] for 9 medusa tokens."
    )

    # model arguments
    # parser.add_argument('--engine_dir', type=str, default='engine_outputs')
    parser.add_argument(
        '--tokenizer_type',
        help=
        'Specify that argument when providing a .model file as the tokenizer_dir. '
        'It allows AutoTokenizer to instantiate the correct tokenizer type.')
    parser.add_argument('--vocab_file',
                        help="Used for sentencepiece tokenizers")
    parser.add_argument('--no_add_special_tokens',
                        dest='add_special_tokens',
                        default=True,
                        action='store_false',
                        help="Whether or not to add special tokens")
    parser.add_argument('--hf_model_dir', '--model_dir', type=str, default=None)
    # parser.add_argument(
    #     '--tokenizer_dir',
    #     default=None,
    #     help='tokenizer path; defaults to hf_model_dir if left unspecified')

    # memory argument
    parser.add_argument(
        '--gpu_weights_percent',
        default=1,
        type=float,
        help=
        'Specify the percentage of weights that reside on GPU instead of CPU and streaming load during runtime.',
    )
    parser.add_argument(
        '--max_tokens_in_paged_kv_cache',
        default=None,
        type=int,
        help=
        'Specify the maximum number of tokens in a kv cache page (only available with cpp session).',
    )
    parser.add_argument(
        '--kv_cache_enable_block_reuse',
        action='store_true',
        help=
        'Enables block reuse in kv cache (only available with cpp session).',
    )
    parser.add_argument(
        '--kv_cache_free_gpu_memory_fraction',
        default=0.9,
        type=float,
        help='Specify the free gpu memory fraction.',
    )
    parser.add_argument(
        '--enable_chunked_context',
        action='store_true',
        help='Enables chunked context (only available with cpp session).',
    )

    # hf model argument (if use hf model)
    parser.add_argument(
        '--hf_data_type',
        '--data_type',
        type=str,
        choices=['fp32', 'fp16', 'bf16', 'float32', 'float16', 'bfloat16'],
        default='fp16',
        help="The data type for hf model.")
    parser.add_argument(
        '--hf_device_map_auto',
        action='store_true',
        help="Use device map 'auto' to load a pretrained HF model. This may "
             "help to test a large model that cannot fit into a singlue GPU.")

    parser.add_argument(
        "--return_all_generated_tokens",
        default=False,
        action="store_true",
        help="if false, return only generated tokens at each streaming step."
             "If true, return the full beams/outputs at each step"
             "Overwritten to True if num_beams>1 and streaming"
             "(only available with cpp session). "
             "WARNING: using this option may increase network usage significantly (quadratically w.r.t output length)."
    )

    return parser


def parse_arguments(args_copy, args=None):
    # see `add_common_args` for extended list of arguments
    parser = argparse.ArgumentParser()
    for key, default_value in args_copy.items():
        parser.add_argument(f'--{key}', type=str, default=default_value,
                            help=f'An optional parameter with default value {default_value}.')
    parser.add_argument('--max_input_length', type=int, default=923)
    parser.add_argument(
        '--input_file',
        type=str,
        help=
        'CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument('--output_csv',
                        type=str,
                        help='CSV file where the tokenized output is stored.',
                        default=None)
    parser.add_argument('--output_npy',
                        type=str,
                        help='Numpy file where the tokenized output is stored.',
                        default=None)
    parser.add_argument(
        '--output_logits_npy',
        type=str,
        help=
        'Numpy file where the generation logits are stored. Use only when num_beams==1',
        default=None)
    parser.add_argument('--output_log_probs_npy',
                        type=str,
                        help='Numpy file where the log_probs are stored',
                        default=None)
    parser.add_argument('--output_cum_log_probs_npy',
                        type=str,
                        help='Numpy file where the cum_log_probs are stored',
                        default=None)
    parser.add_argument(
        '--run_profiling',
        default=False,
        action='store_true',
        help="Run several 10 iterations to profile the inference latencies.")

    parser = add_common_args(parser)

    return parser.parse_args(args=args)


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
    if output_csv is None and output_npy is None:
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

                for item in outputs1:
                    streamer.put(torch.tensor([item]))


def get_trt_llm_base_args(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    # different handling if encoder-decoder models
    is_enc_dec = {
                     name
                     for name in os.listdir(args.engine_dir)
                     if os.path.isdir(os.path.join(args.engine_dir, name))
                 } == {'encoder', 'decoder'}
    if is_enc_dec:
        logger.warning(
            "This path is an encoder-decoder model. Using different handling.")
        assert not args.use_py_session, "Encoder-decoder models don't have a unified python runtime, please use its own examples/enc_dec/run.py instead."

    model_name, model_version = read_model_name(
        args.engine_dir) if not is_enc_dec else ("", "")
    if args.tokenizer_dir is None and model_name in DEFAULT_HF_MODEL_DIRS:
        logger.warning(
            "tokenizer_dir is not specified. Try to infer from model_name, but this may be incorrect."
        )
        args.tokenizer_dir = DEFAULT_HF_MODEL_DIRS[model_name]

    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        vocab_file=args.vocab_file,
        model_name=model_name,
        model_version=model_version,
        tokenizer_type=args.tokenizer_type,
    )

    stop_words_list = None
    if args.stop_words:
        stop_words_list = tensorrt_llm.runtime.decode_words_list(
            args.stop_words, tokenizer)

    bad_words_list = None
    if args.bad_words:
        bad_words_list = tensorrt_llm.runtime.decode_words_list(
            args.bad_words, tokenizer)

    prompt_template = None
    if args.use_prompt_template and model_name in DEFAULT_PROMPT_TEMPLATES:
        prompt_template = DEFAULT_PROMPT_TEMPLATES[model_name]

    return args, is_enc_dec, runtime_rank, tokenizer, end_id, pad_id, stop_words_list, bad_words_list, model_name, model_version, prompt_template


class trtLLMQwen(BaseEngine):
    def __init__(self, args_copy, args, trt_llm_args, is_enc_dec, runtime_rank, tokenizer, end_id, pad_id,
                 stop_words_list, bad_words_list, model_name, model_version, prompt_template,
                 model_args: "ModelArguments" = None,
                 data_args: "DataArguments" = None,
                 finetuning_args: "FinetuningArguments" = None,
                 generating_args: "GeneratingArguments" = None
                 ):

        self.runner_kwargs = {
            'engine_dir': args_copy.engine_dir,
            'lora_dir': None,
            'rank': 0,
            'debug_mode': False,
            'lora_ckpt_source': 'hf',
            'gpu_weights_percent': 1,
            'is_enc_dec': False,
            'max_batch_size': 1,
            'max_input_len': 1,
            'max_output_len': 1,
            'max_beam_width': 1,
            'max_attention_window_size': None,
            'sink_token_length': None,
            'max_tokens_in_paged_kv_cache': None,
            'kv_cache_enable_block_reuse': False,
            'kv_cache_free_gpu_memory_fraction': 0.9,
            'enable_chunked_context': False
        }
        self.args = args
        self.is_enc_dec = is_enc_dec
        self.runtime_rank = runtime_rank
        self.tokenizer = tokenizer
        self.end_id = end_id
        self.pad_id = pad_id
        self.stop_words_list = stop_words_list
        self.bad_words_list = bad_words_list
        self.model_name = model_name
        self.model_version = model_version
        self.prompt_template = prompt_template

        self.can_generate = True
        self.tokenizer.padding_side = "left"
        self.template = get_template_and_fix_tokenizer(self.tokenizer, 'qwen')

        self.runner = ModelRunnerCpp.from_dir(**self.runner_kwargs)


    @staticmethod
    def run_update_trt_llm_args(self, args, tokenizer, prompt_template, model_name, model_version, is_enc_dec,
                                runtime_rank,
                                stop_words_list, pad_id, end_id, bad_words_list,
                                messages: Sequence[Dict[str, str]], template: "Template", system: Optional[str] = None,
                                tools: Optional[str] = None):
        paired_messages = messages + [{"role": "assistant", "content": ""}]
        prompt_ids, _ = template.encode_oneturn(
            tokenizer=tokenizer, messages=paired_messages, system=system, tools=tools
        )

        batch_input_ids = torch.tensor([prompt_ids], dtype=torch.int32)

        encoder_input_ids = None
        decoder_input_ids = None
        if is_enc_dec:
            encoder_input_ids = batch_input_ids
            decoder_start_token_id = read_decoder_start_token_id(
                os.path.join(args.engine_dir, "decoder"))
            decoder_input_ids = [
                torch.tensor([decoder_start_token_id], dtype=torch.int32)
                for _ in batch_input_ids
            ]

        input_lengths = [x.size(0) for x in decoder_input_ids
                         ] if is_enc_dec else [x.size(0) for x in batch_input_ids]
        encoder_input_lengths = [x.size(0)
                                 for x in encoder_input_ids] if is_enc_dec else None

        if not supports_inflight_batching(
                os.path.join(args.engine_dir, "decoder") if is_enc_dec else args.
                        engine_dir):
            logger.warning(
                "The given engine does not support in-flight batching, fallback to python session"
            )
            args.use_py_session = True

        if not PYTHON_BINDINGS and not args.use_py_session:
            logger.warning(
                "Python bindings of C++ session is unavailable, fallback to Python session."
            )
            args.use_py_session = True
        if args.debug_mode and not args.use_py_session:
            logger.warning(
                "Debug mode is not supported in C++ session for now, fallback to Python session."
            )
            args.use_py_session = True
        if args.return_all_generated_tokens and args.use_py_session:
            raise ValueError(
                "Returning all the generated tokens at each step is not supported in the Python session, use C++ session instead."
            )
        if (not args.return_all_generated_tokens) and args.streaming and (
                args.num_beams > 1):
            logger.warning(
                "Setting return_all_generated_tokens to True since streaming AND beam search are done simultaneously. "
                "Returning the full beams at each streaming step is needed because beam search + streaming can change previous outputs. "
                "WARNING: using this option may increase network usage significantly (quadratically w.r.t output length)."
            )
            args.return_all_generated_tokens = True
        # runner_cls = ModelRunner if args.use_py_session else ModelRunnerCpp
        runner_kwargs = dict(
            engine_dir=args.engine_dir,
            lora_dir=args.lora_dir,
            rank=runtime_rank,
            debug_mode=args.debug_mode,
            lora_ckpt_source=args.lora_ckpt_source,
            gpu_weights_percent=args.gpu_weights_percent,
        )
        if not args.use_py_session:
            runner_kwargs.update(is_enc_dec=is_enc_dec)
        if args.medusa_choices is not None:
            args.medusa_choices = ast.literal_eval(args.medusa_choices)
            assert args.temperature == 1.0, "Medusa should use temperature == 1.0"
            assert args.num_beams == 1, "Medusa should use num_beams == 1"
            runner_kwargs.update(medusa_choices=args.medusa_choices)

        return args, is_enc_dec, runtime_rank, tokenizer, batch_input_ids, encoder_input_ids, decoder_input_ids, input_lengths, \
            end_id, pad_id, stop_words_list, bad_words_list

    @staticmethod
    def run_trt_llm(**kwargs):
        runner_trt_llm = kwargs.get('runner')
        args = kwargs.get('args')
        is_enc_dec = kwargs.get('is_enc_dec')
        runtime_rank = kwargs.get('runtime_rank')
        tokenizer = kwargs.get('tokenizer')
        batch_input_ids = kwargs.get('batch_input_ids')
        encoder_input_ids = kwargs.get('encoder_input_ids')
        decoder_input_ids = kwargs.get('decoder_input_ids')
        input_lengths = kwargs.get('input_lengths')
        end_id = kwargs.get('end_id')
        pad_id = kwargs.get('pad_id')
        stop_words_list = kwargs.get('stop_words_list')
        bad_words_list = kwargs.get('bad_words_list')
        streamer =  kwargs.get('streamer')

        # 把batch input ids去掉
        if streamer is not None:
            streamer.put(batch_input_ids.cpu())

        # 相差结果大不大，
        with torch.no_grad():
            outputs = runner_trt_llm.generate(
                batch_input_ids=decoder_input_ids
                if is_enc_dec else batch_input_ids,
                encoder_input_ids=encoder_input_ids if is_enc_dec else None,
                max_new_tokens=args.max_output_len,
                max_attention_window_size=args.max_attention_window_size,
                sink_token_length=args.sink_token_length,
                end_id=end_id,
                pad_id=pad_id,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                early_stopping=args.early_stopping,
                repetition_penalty=args.repetition_penalty,
                presence_penalty=args.presence_penalty,
                frequency_penalty=args.frequency_penalty,
                stop_words_list=stop_words_list,
                bad_words_list=bad_words_list,
                output_cum_log_probs=(args.output_cum_log_probs_npy != None),
                output_log_probs=(args.output_log_probs_npy != None),
                random_seed=args.random_seed,
                lora_uids=args.lora_task_uids,
                prompt_table=args.prompt_table_path,
                prompt_tasks=args.prompt_tasks,
                streaming=args.streaming,
                output_sequence_lengths=True,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                return_dict=True,
                medusa_choices=args.medusa_choices,
                return_all_generated_tokens=args.return_all_generated_tokens)
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
        if args.output_cum_log_probs_npy != None:
            cum_log_probs = outputs['cum_log_probs']
        if args.output_log_probs_npy != None:
            log_probs = outputs['log_probs']

        print_output(tokenizer,
                     output_ids,
                     input_lengths,
                     sequence_lengths,
                     output_csv=args.output_csv,
                     output_npy=args.output_npy,
                     context_logits=context_logits,
                     generation_logits=generation_logits,
                     output_logits_npy=args.output_logits_npy,
                     cum_log_probs=cum_log_probs,
                     log_probs=log_probs,
                     output_cum_log_probs_npy=args.output_cum_log_probs_npy,
                     output_log_probs_npy=args.output_log_probs_npy, streamer = streamer)

        if streamer is not None:
            streamer.end()


    @staticmethod
    @torch.inference_mode()
    def _stream_chat(
            self,
            args, is_enc_dec, runtime_rank, tokenizer, end_id, pad_id,
            stop_words_list, bad_words_list, model_name, model_version, prompt_template,
            template,
            messages: Sequence[Dict[str, str]],
            generating_args: Dict[str, Any],
            system: Optional[str] = None,
            tools: Optional[str] = None,
            input_kwargs: Optional[Dict[str, Any]] = {},
    ) -> Callable[[], str]:
        args, is_enc_dec, runtime_rank, tokenizer, batch_input_ids, encoder_input_ids, decoder_input_ids, input_lengths, end_id, pad_id, stop_words_list, bad_words_list \
            = trtLLMQwen.run_update_trt_llm_args(self, args, tokenizer,
                                                 prompt_template,
                                                 model_name,
                                                 model_version,
                                                 is_enc_dec,
                                                 runtime_rank,
                                                 stop_words_list,
                                                 pad_id, end_id,
                                                 bad_words_list, messages, template)


        gen_kwargs = {
            "runner": self.runner,
            "args": args,
            "is_enc_dec": is_enc_dec,
            "runtime_rank": runtime_rank,
            "tokenizer": tokenizer,
            "batch_input_ids": batch_input_ids,
            "encoder_input_ids": encoder_input_ids,
            "decoder_input_ids": decoder_input_ids,
            "input_lengths": input_lengths,
            "end_id": 151645,
            "pad_id": 151645,
            "stop_words_list": stop_words_list,
            "bad_words_list": bad_words_list,
        }
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer
        thread = Thread(target=self.run_trt_llm, kwargs=gen_kwargs, daemon=True)
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
            self,
            self.args, self.is_enc_dec, self.runtime_rank,
            self.tokenizer, self.end_id, self.pad_id, self.stop_words_list,
            self.bad_words_list,
            self.model_name,
            self.model_version,
            self.prompt_template,
            self.template,
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
        gen_kwargs, prompt_length = trtLLMQwen._process_args(
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
