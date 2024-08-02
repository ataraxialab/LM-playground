from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class GeneratingArguments:
    r"""
    Arguments pertaining to specify the decoding parameters.
    """

    do_sample: bool = field(
        default=True,
        metadata={"help": "Whether or not to use sampling, use greedy decoding otherwise."},
    )
    temperature: float = field(
        default=0.95,
        metadata={"help": "The value used to modulate the next token probabilities."},
    )
    top_p: float = field(
        default=0.7,
        metadata={
            "help": "The smallest set of most probable tokens with probabilities that add up to top_p or higher are kept."
        },
    )
    top_k: int = field(
        default=50,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k filtering."},
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."},
    )
    max_length: int = field(
        default=512,
        metadata={"help": "The maximum length the generated tokens can have. It can be overridden by max_new_tokens."},
    )
    max_new_tokens: int = field(
        default=512,
        metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."},
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."},
    )
    length_penalty: float = field(
        default=1.0,
        metadata={"help": "Exponential penalty to the length that is used with beam-based generation."},
    )

    length_penalty: float = field(
        default=1.0,
        metadata={"help": "Exponential penalty to the length that is used with beam-based generation."},
    )
    early_stopping: int = field(
        default=1,
        metadata={"help": "Use early stopping if num_beams > 1 1 for early-stopping, 0 for non-early-stopping other values for stopping by length"},
    )
    random_seed: int = field(
        default=0, 
        metadata={"help": "Random seed for generation."}
    )
    

## QA generation args
    filepath: str = field(
        default="",
        metadata={"help": "The file path of raw input lines."},
    )
    prompt_path: str = field(
        default=True,
        metadata={"help": "The path of generation prompt"},
    )

## trt-llm generation args
    trt_max_attention_window_size: int = field(
        default=None,
        metadata={"help":"The attention window size that controls the sliding window attention / cyclic kv cache behavior"},
    )
    trt_sink_token_length: int = field(
        default=None,
        metadata={"help": "The sink token length."},
    )
    trt_presence_penalty: float = field(
        default=0.0,
    )
    trt_frequency_penalty: float = field(
        default=0.0,
    )
    trt_output_cum_log_probs_npy: str = field(
        default=None,
        metadata={"help": "Numpy file where the cum_log_probs are stored."},
    )
    trt_output_log_probs_npy: str = field(
        default=None,
        metadata={"help": "Numpy file where the log_probs are stored."},
    )
    trt_lora_task_uids: str = field(
        default=None,
        metadata={"help": "The list of LoRA task uids; use -1 to disable the LoRA module"},
    )
    trt_prompt_table: str = field(
        default=None,
        metadata={"help": "Path to .npy file, exported by nemo_prompt_convert.py."},
    )
    trt_prompt_tasks: str = field(
        default=None,
        metadata={"help": "Comma-separated list of tasks for prompt tuning, e.g., 0,3,1,0"},
    )
    trt_streaming: bool = field(
        default=False,
        metadata={"help": "Whether to stream the output."}
    )
    trt_no_repeat_ngram_size: int = field(
        default=None,
    )
    trt_medusa_choices: str = field(
        default=None,
        metadata={"help": "Medusa choice to use, if not none, will use Medusa decoding."
        "   E.g.: [[0, 0, 0, 0], [0, 1, 0], [1, 0], [1, 1]] for 9 medusa tokens."}
    )
    trt_num_beams: int = field(
        default=1,
        metadata={"help": "Use beam search if num_beams > 1"}
    )
    trt_max_tokens_in_paged_kv_cache : int = field(
        default=None,
        metadata={"help": "Specify the maximum number of tokens in a kv cache page (only available with cpp session)."}
    )
    trt_kv_cache_enable_block_reuse: bool = field(
        default=False,
        metadata={"help": "Enables block reuse in kv cache (only available with cpp session)."},
    )
    trt_kv_cache_free_gpu_memory_fraction: float = field(
        default=0.9,
        metadata={"help": "Specify the free gpu memory fraction."},
    )
    trt_enable_chunked_context: bool = field(
        default=False,
        metadata={"help": "Enables chunked context (only available with cpp session)."},
    )
    trt_return_all_generated_tokens: bool = field(
        default=False,
        metadata={"help": "If false, return only generated tokens at each streaming step."
             "If true, return the full beams/outputs at each step"
             "Overwritten to True if num_beams>1 and streaming"
             "(only available with cpp session). "
             "WARNING: using this option may increase network usage significantly (quadratically w.r.t output length)."},
    )
    trt_debug_mode: bool = field(
        default=False,
        metadata={"help": "Whether or not to turn on the debug mode"},
    )




    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)
        if args.get("max_new_tokens", -1) > 0:
            args.pop("max_length", None)
        else:
            args.pop("max_new_tokens", None)
        return args
