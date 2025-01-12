from typing import Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class DataArgs:
    data_path: str = "./datas/train"
    data_type: str = "train"
    task_type: str = "all"



@dataclass
class TrainArgs:
    # training_config
    lr: float = 5e-6
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.01

    batch_size: int = 2
    epochs: int = 3
    lr_scheduler: str = "cosine"
    warm_up: bool = True
    # 最好应该是根据数据集的大小来判断，这些参数需要修改
    warmup_steps: int = 100
    lr_min: float = 1e-7
    max_steps: int = 1000000
    warmup_ratio: float = 0.02
    seed: int = 567
    num_steps: int = field(
        default=1000000,
        metadata={"help": "the number of steps for training,useless"},
    )

    # fsdp config
    mixed_precision: bool = field(
        default=True,
        metadata={"help": "whether to use mixed precision training,default is True"},
    )

    wrap_block: str = field(
        default="transformer",
        metadata={"help": "the type of fsdp wrapper, e.g. ['transformer', 'linear']"},
    )
    wrapper_type: str = field(
        default="transformer",
        metadata={
            "help": "the type of fsdp wrapper, e.g. ['size_based', 'transformer']"
        },
    )

    min_num_params: float = 1e6

    sharding_strategy: str = field(
        default="fsdp", metadata={"choices": "['fsdp','hspd','ddp']"}
    )

    low_cpu_fsdp: bool = field(
        default=True,
        metadata={"help": "whether to use low cpu fsdpL,default is False"},
    )

    fsdp_activation_checkpointing: bool = field(
        default=True,
        metadata={"help": "whether to use activation checkpointing,default is False"},
    )
    selective_checkpointing: str = field(
        default="1/3",
        metadata={
            "help": "p:the number of layers to apply selective checkpointing,default is 0,such as '1/3'"
        },
    )

    use_torch_compile: bool = field(
        default=True,
        metadata={
            "help": "whether to use torch.jit.script to compile the model,default is True"
        },
    )

    use_profiler: bool = field(
        default=True,
        metadata={"help": "whether to use profiler,default is False"},
    )
    profile_traces:str = field(
        default="./logs/profiler",
        metadata={"help": "the type of profile traces,default is 'default'"},
    )

    profiler_rank0_only: bool = field(
        default=True,
        metadata={"help": "whether to use profiler only on rank0,default is True"},
    )

    grad_clip_thresh: float = field(
        default=2.0,
        metadata={"help": "the threshold for gradient clipping,default is 1.0"},
    )
    tracker: str = field(
        default="wandb",
        metadata={"help": "whether to use tracker,default is False"},
    )
    tracker_dir: str = field(
        default="./logs/tracker",
        metadata={"help": "the path to save the tracker logs"},
    )
    tracker_project_name: str = field(
        default="mllm",
        metadata={"help": "the project name for tracker"},
    )

    use_lora: bool = field(default=True, metadata={"help": "whether to use loram"})
    # 和 dataloader 的shuffle 是相反的
    shuffle: bool = field(
        default=True,
        metadata={"help": "whether to shuffle the distributed sampler"},
    )

    num_workers: int = field(
        default=2, metadata={"help": "the number of workers for dataloader"}
    )

    ckpt_save_path: str = field(
        default="./checkpoints/",
        metadata={"help": "the path to save the checkpoints"},
    )

    finetune: bool = field(
        default=True, metadata={"help": "whether to finetune the model"}
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "the number of gradient accumulation steps,default is 1"},
    )

    report_interval: int = field(
        default=100,
        metadata={"help": "the interval for reporting the training status"},
    )

    checkpoint_interval: int = field(
        default=1000,
        metadata={"help": "the interval for saving the checkpoints"},
    )

    save_only_rank0: bool = field(
        default=False,
        metadata={"help": "whether to save only on rank0,default is True"},
    )



# 采用 typing 中的，必须添加 List[int]
@dataclass
class ModelArgs:
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    cache_dir :str= "./huggingface/hub"
    device_map:str = field(
        default="cpu",
        metadata={"help": "the device map for fsdp"},
    )

@dataclass
class OptunaArgs:
    n_trials: int = 20
    timeout: int = 600
    n_jobs: int = 1
    direction: str = "minimize"
    study_name: str = "mllm_hyper_search"
    search_lr_min: float = 1e-6
    search_lr_max: float = 1e-5
    epochs_max: int = 10
    storage: str = "sqlite:///./logs/hyper_search/mllm.db"
    load_if_exists: bool = True
    optuna_seed: int = 678
    low_rank_max: int = 16
    low_rank_min: int = 4
    low_rank_step: int = 4
    # search_lora_alpha: int = 1
    search_lora_alpha_min: int = 4
    search_lora_alpha_max: int = 16
    search_lora_alpha_step: int = 4


@dataclass
class PeftArgs:
    low_rank: int = 8
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    peft_type: str = field(
        default="lora", metadata={"choices": "['lora', 'lora_bias']"}
    )
    lora_alpha: int = 8
    bias: str = "none"
    adapter_name: str = "sft_qwen_vl"



@dataclass
class FsdpEvaluationArgs:
    wrap_block:str = "transformer"
    mixed_precision:bool = True
    wrapper_type:str = "transformer"
    sharding_strategy:str = "fsdp"
    low_cpu_fsdp:bool = True
    fsdp_activation_checkpointing:bool = True
    selective_checkpointing:str = "1/3"
    use_torch_compile:bool = True

    tracker:str = "wandb"
    tracker_dir:str = "./logs/tracker"
    tracker_project_name:str = "mllm"
    
    model_name:str = "Qwen/Qwen2-VL-2B-Instruct"
    cache_dir :str= "./huggingface/hub"
    
    data_path:str = "./datas/test"
    task_type:str = "all"
    data_type:str = "test"
    
    shuffle:bool = False
    seed:int = 567
    batch_size:int = 1
    num_workers:int = 2
    

    use_profiler:bool = True
    profile_traces:str = "./logs/profiler"
    
    use_lora:bool = False
    low_rank:int = 8
    target_modules:List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    peft_type:str = "lora"
    lora_alpha:int = 8
    bias:str = "none"
    adapter_name:str = "sft_qwen_vl"
    
    ckpt_save_path:str='./checkpoints/'
    load_file_name:str = "best.ckpt"
    profiler_rank0_only:bool = True
    
    sft:bool = False