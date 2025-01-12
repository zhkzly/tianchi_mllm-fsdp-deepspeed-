import math
import os

import torch
import torch.optim as optim
from dataclasses import asdict

# from fms.models.llama import LLaMA, LLaMABlock
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# from torch.optim.lr_scheduler import LambdaLR
from rich import print
from src.utils.fms_fsdp.utils.checkpointing_utils import Checkpointer

# from src.utils.fms_fsdp.utils.config_utils import get_model_config
import math
from src.utils.fms_fsdp.utils.train_utils import (
    get_policies,
    get_profiler,
    setup,
    setup_environ_flags,
    train,
    get_wrap_block,
    validate_fn,
)
from src.utils.utils import get_model
from src.hyper_search.configs import ModelArgs, TrainArgs, PeftArgs, DataArgs
from transformers import HfArgumentParser
from peft import LoraConfig, get_peft_model

import optuna

from src.utils.fms_fsdp.utils.dummy_dataloader import get_dataloaders
from src.utils.fms_fsdp.utils.dummy_datasets import CustomSftDataset, DataSftCollator

from transformers import AutoProcessor
from src.utils.utils import get_optimizer, get_scheduler
from tqdm import tqdm


from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

import os

# import shutil
import time

# from pathlib import Path

import torch
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load,
    save,
    load_sharded_optimizer_state_dict,
)

# from torch.distributed._shard.checkpoint import (
#     FileSystemReader,
#     FileSystemWriter,
#     load,
#     save,
# )


def optuna_main(
    train_args: TrainArgs,
    model_args: ModelArgs,
    data_args: DataArgs,
    peft_args: PeftArgs,
    trial,
):
    # get configs

    # ensure reproducibility
    torch.cuda.manual_seed(train_args.seed)
    torch.manual_seed(train_args.seed)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"--> running with these configs train_args: {train_args}\n")
        print(f"--> running with these configs model_args: {model_args}\n")
        print(f"--> running with these configs data_args: {data_args}\n")
        print(f"--> running with these configs peft_args: {peft_args}\n")

    # some setups
    # 不可以重复dist.init， 否则会报错
    # setup()
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    setup_environ_flags()

    # get policy
    # 给哪个 module 进行封装，在这里采用尺寸进行判断，
    # such as transformer_block
    block = get_wrap_block(train_args, rank)
    (
        mixed_precision_policy,
        wrapping_policy,
        sharding_strategy_policy,
        apply_selective_ac,
        param_init_fn,
    ) = get_policies(train_args, rank, block)

    # get fms model
    # llama_config = get_model_config(model_args.model_name)
    if train_args.low_cpu_fsdp:
        print(f"--> using Lora for low cpu fsdp and low rank...")
        # 这里会遇到加载的问题，微调模型，所以就必须保留原始的参数，而meta设备上的虽然分片，减少内存占用
        # 但是meta上的参数都是被随机化的，无法保留自己想要的
        # with torch.device("meta"):
        #     model = get_model(model_args.model_name)
        #     lora_config = LoraConfig(
        #         r=peft_args.low_rank,
        #         target_modules=peft_args.target_modules,
        #         peft_type=peft_args.task_type,
        #         lora_alpha=peft_args.lora_alpha,
        #         bias=peft_args.bias,
        #     )
        # model = get_peft_model(
        #     model=model, peft_config=lora_config, adapter_name=peft_args.adapter_name
        # )
        # float32,cpu
        model = get_model(model_args)
        if train_args.use_lora:
            lora_config = LoraConfig(
                r=peft_args.low_rank,
                target_modules=peft_args.target_modules,
                peft_type=peft_args.peft_type,
                lora_alpha=peft_args.lora_alpha,
                bias=peft_args.bias,
            )
            model = get_peft_model(
                model=model,
                peft_config=lora_config,
                adapter_name=peft_args.adapter_name,
            )
            for name, param in model.named_parameters():
                print(f"Parameter {name} has dtype: {param.dtype}")
                break

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> model has {total_params / 1e6} Million params\n")

    # get data loader
    if rank == 0:
        print("Constructing datasets...")
        print(f"--> using data_path: {data_args.data_path}")
        print(f"loading qwen2 processor...")

    preprocessor = AutoProcessor.from_pretrained(model_args.model_name)
    # os.path.join(data_args.data_path,data_type, ".json")
    train_dataset = CustomSftDataset(
        preprocessor=preprocessor,
        data_path=data_args.data_path,
        data_type=data_args.data_type,
        task_type=data_args.task_type,
    )
    val_dataset = CustomSftDataset(
        preprocessor=preprocessor,
        data_path=data_args.data_path,
        data_type="val",
        task_type="all",
    )
    data_collator = DataSftCollator(preprocessor=preprocessor)
    train_sampler, train_loader, val_loader = get_dataloaders(
        train_dataset,
        val_dataset,
        world_size=world_size,
        local_rank=rank,
        shuffle=train_args.shuffle,
        seed=train_args.seed,
        collator=data_collator,
        batch_size=train_args.batch_size,
        num_workers=train_args.num_workers,
    )
    train_args.max_steps = (
        train_args.epochs
        * len(train_dataset)
        // (train_args.batch_size * train_args.gradient_accumulation_steps)
    )
    train_args.warmup_steps = math.ceil(train_args.warmup_ratio * train_args.max_steps)
    dist.barrier()
    if rank == 0:
        print("Datasets constructed!")
        print("successfully constructed data loaders!")

        # FSDP,应该是在这里进行了模型参数的初始化，也就是放置到了实际的gpu上

        if rank == 0:
            print(f"--> initializing fsdp model...")
    model = FSDP(
        model,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=sharding_strategy_policy,
        # 必须和compile同时使用，否则会报错,
        use_orig_params=train_args.use_torch_compile,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        param_init_fn=None,
        sync_module_states=False,
    )

    # fsdp activation checkpointing
    if train_args.fsdp_activation_checkpointing:
        if rank == 0:
            print(f"--> applying FSDP activation checkpointing...")
        apply_selective_ac(model, p=train_args.selective_checkpointing)

    # torch compile
    # if train_args.use_torch_compile:
    #     if rank == 0:
    #         print(f"--> enabling torch compile...")
    #     # the default accumulated_cache_size_limit=64 is not enough for 70b model, so we make it 128 here
    #     torch._dynamo.config.accumulated_cache_size_limit = 128
    #     model = torch.compile(model)

    # Optimizer
    # optimizer = optim.AdamW(
    #     model.parameters(),
    #     lr=train_args.learning_rate,
    #     betas=(0.9, 0.95),
    #     weight_decay=0.1,
    # )
    optimizer = get_optimizer(model, train_args)

    # optionally load from checkpoint (when continue pretraining)
    checkpointer = Checkpointer(
        train_args.ckpt_save_path, train_args.sharding_strategy, rank, local_rank
    )

    scheduler = get_scheduler(optimizer, train_args)

    # profiler
    profiler = get_profiler(train_args, rank)

    # Train
    if rank == 0:
        print(f"Training for {train_args.num_steps} steps...")

    pbar = tqdm(
        range(train_args.epochs),
        total=train_args.epochs,
        colour="blue",
        desc=f"traing epochs",
        disable=(rank != 0),
    )
    if rank == 0:
        if train_args.tracker:
            tracker_dir = train_args.tracker_dir
            project_name = train_args.tracker_project_name
            run_id = f"{trial.number}" if trial else None

            if train_args.tracker == "wandb":
                try:
                    import wandb  # type: ignore
                except ImportError:
                    raise ImportError(
                        "tracker is set to wandb but wandb is not installed."
                    )
                if rank == 0:
                    print(f"--> wandb is enabled!")
                    try:
                        wandb.init(
                            project=project_name,
                            dir=tracker_dir,
                            resume="allow",
                            id=run_id,
                            mode="offline",
                        )
                    except wandb.errors.UsageError:
                        raise ValueError(
                            "wandb failed to init, did you pass your wandb api key via WANDB_API_KEY?"
                        )
                    wandb.config = asdict(train_args)
    for epoch in pbar:
        val_to_track = train(
            train_args,
            model,
            local_rank,
            rank,
            train_loader,
            optimizer,
            scheduler,
            profiler,
            checkpointer=checkpointer,
            start_step=0,
            epoch=epoch,
            train_sampler=train_sampler,
            trial=trial,
        )
        if train_args.tracker:
            wandb.log(val_to_track, step=epoch)
        # epoch=0
        validate_loss = validate_fn(
            model=model,
            val_loader=val_loader,
            train_args=train_args,
            local_rank=local_rank,
            rank=rank,
            epoch=epoch,
            tracker_fn=wandb.log,
        )
        trial.report(validate_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    if train_args.save_only_rank0:
        checkpointer.save_single_file(
            step=trial.number, model=model, is_compiled=train_args.use_torch_compile
        )
        dist.barrier()
    else:
        checkpointer.save(model=model, optimizer=optimizer, step=trial.number)
        dist.barrier()

    return validate_loss


# 该类主要是用来加载和保存模型的参数的，包括单个文件和分片文件。
# 单个文件就是指模型的参数全部保存在一个文件中，而分片文件就是指模型的参数被分片保存在多个文件中。
# 单个文件加载和保存的过程比较简单，而分片文件加载和保存的过程则需要使用到torch.distributed._shard.checkpoint模块。
# 该模块提供了两个类FileSystemReader和FileSystemWriter，用于读取和写入分片文件。
# 该模块还提供了两个函数load和save，用于加载和保存模型的参数。
class Checkpointer:
    """
    Manages the checkpoint directory. Saves new checkpoints and deletes old ones after the specified number are written.
    Also handles loading and saving of checkpoints in sharded and unsharded formats.
    Assumes model and optimizer inputs are in FSDP.
    ...
    Args
    ----
    ckpdir : str
        Absolute path to desired save location. Creates a new 'checkpoints/' subfolder at that location.
    n_to_save : int
        Number of volatile checkpoints to maintain at any given time.
    parallel_mode : str
        Write sharded folder ckps (when sharded: 'fsdp' or 'hsdp') or unsharded file ckps (when sharded: 'ddp')
    report_fn : Callable or None
        Optional function for reporting or logging status updates. Expected to handle arbitrary *args, **kwargs.
        Defaults to self._selective_print().
    model_auto_placement : bool
        Optional; If True, auto detect GPU device to move model to, as set in device mesh init

    Methods
    -------
    save : keyword args -> str | None
        Saves dictionary of keyword arg key/value pairs to specified checkpoint directory, deleting old checkpoints
        as necessary. If a checkpoint is deleted, returns the filename of that checkpoint.
    load :
        See docstring for individual function below
    """

    def __init__(
        self,
        ckpdir,
        parallel_mode,
        rank,
        local_rank,
        model_auto_placement=False,
    ):
        self.rank = rank
        self.local_rank = local_rank
        self.ckp_path = os.path.join(ckpdir, "checkpoints/")
        os.makedirs(self.ckp_path, exist_ok=True)
        self.p_mode = parallel_mode
        assert parallel_mode in ["fsdp", "hsdp", "ddp"]

        self.model_auto_placement = model_auto_placement

    def report(self, output):
        if self.rank == 0:
            print(output)

    def load(
        self,
        model,
        optimizer=None,
        file_name=None,
    ):
        """
        Handle checkpoint loading for model/optimizer/dataloader from given path, according to arguments.
        Defaults to save path for locating an appropriate checkpoint. If a path is provided, will use
        it only if no appropriate checkpoint is found in the save path (in which case it's a job restart).
        Reset_stepcount manually resets optimizer and dataloader states, and stat tracking.
        Strict determines whether to use strict loading or not FOR SINGLEFILE LOADING ONLY.
        Returns model, optimizer, dataloader, current step, and current tokens seen.
        """
        model_load_time = time.time()
        # Load model
        load_path = os.path.join(self.ckp_path, file_name)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict = model.state_dict()
            model_ckp = {"model_state": state_dict}
            load(
                state_dict=model_ckp,
                storage_reader=FileSystemReader(load_path),
                planner=DefaultLoadPlanner(),
            )
            model.load_state_dict(model_ckp["model_state"])
        if self.model_auto_placement:
            model.to("cuda")
        else:
            model.to(self.local_rank)
        self.report(model_load_time=time.time() - model_load_time)
        # Load optimizer
        if optimizer is not None:
            optim_load_time = time.time()
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                optim_state = load_sharded_optimizer_state_dict(
                    model_state_dict=model.state_dict(),
                    optimizer_key="optimizer_state",
                    storage_reader=FileSystemReader(load_path),
                )
            flattened_osd = FSDP.optim_state_dict_to_load(
                model, optimizer, optim_state["optimizer_state"]
            )
            optimizer.load_state_dict(flattened_osd)
            self.report(optimizer_load_time=time.time() - optim_load_time)
            return model, optimizer
        else:
            return model

    def save(
        self,
        step,
        model,
        optimizer,
        **kwargs,
    ):
        # Note: metadata kwargs cannot contain any of:
        # (step, model, optimizer, dataloader)
        rank = self.rank
        save_time = time.time()
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            model_state = model.state_dict()
            optim_state = FSDP.sharded_optim_state_dict(model, optimizer)

        save_name = os.path.join(self.ckp_path, "step_" + str(step) + "_ckp")
        state_dict = {"model_state": model_state, "optimizer_state": optim_state}
        if self._do_save(rank, self.local_rank):
            self._write(state_dict, model.process_group, save_name, rank)
        if rank == 0:
            metadata = kwargs
            metadata["step"] = step
            torch.save(metadata, os.path.join(save_name, "metadata.pth"))
        self.report(
            f"Checkpoint saved in {save_name}", model_save_time=time.time() - save_time
        )

        return

    def _write(self, state_dict, process_group, save_name, rank):
        os.makedirs(save_name, exist_ok=True)
        writer = FileSystemWriter(save_name, single_file_per_rank=True)
        if state_dict is not None:
            save(
                state_dict=state_dict,
                storage_writer=writer,
                process_group=process_group,
                planner=DefaultSavePlanner(),
            )

    def save_single_file(
        self,
        step,
        model,
        is_compiled=False,
        **kwargs,
    ):
        # Note: metadata kwargs cannot contain any of:
        # (step, model)
        save_name = os.path.join(self.ckp_path, "step_" + str(step) + "_ckp.pth")
        save_time = time.time()
        # model的名称中含有fsdp的封装名称：https://github.com/AnswerDotAI/fsdp_qlora/blob/main/train.py
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            if is_compiled:
                model_state = model._orig_mod.state_dict()
            else:
                model_state = model.state_dict()
        if self.rank == 0:
            metadata = kwargs
            metadata["step"] = step
            metadata["model_state"] = model_state
            torch.save(metadata, save_name)
        self.report("Checkpoint written", model_save_time=time.time() - save_time)

        return

    # 没有去实现
    def load_single_file(
        self,
        model,
        optimizer,
        path="",
        reset_stepcount=False,
        strict=True,
        is_compiled=False,
    ):
        checkpoint_data = torch.load(load_path, map_location="cpu")
        if is_compiled:
            model._orig_mod.load_state_dict(
                checkpoint_data.get("model_state"), strict=strict
            )
        else:
            model.load_state_dict(checkpoint_data.get("model_state"), strict=strict)
        if self.model_auto_placement:
            model.to("cuda")
        else:
            model.to(self.local_rank)
        return model


if __name__ == "__main__":
    # fire.Fire(main)
    train_args, data_args, peft_args, data_args = HfArgumentParser(
        (ModelArgs, TrainArgs, PeftArgs, DataArgs)
    ).parse_args_into_dataclasses()
    optuna_main(
        train_args=train_args,
        data_args=data_args,
        peft_args=peft_args,
        trial=tiral,
    )
