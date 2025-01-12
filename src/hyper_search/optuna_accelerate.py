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
    get_wrap_block,
    validate_fn,
)
from src.utils.fms_fsdp.utils.train_utils_accelerate import train

from accelerate import Accelerator
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
import datetime

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
    accelerator=Accelerator()
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

    # # get policy
    # # 给哪个 module 进行封装，在这里采用尺寸进行判断，
    # # such as transformer_block
    # block = get_wrap_block(train_args, rank)
    # (
    #     mixed_precision_policy,
    #     wrapping_policy,
    #     sharding_strategy_policy,
    #     apply_selective_ac,
    #     param_init_fn,
    # ) = get_policies(train_args, rank, block)

    # get fms model
    # llama_config = get_model_config(model_args.model_name)
    if train_args.low_cpu_fsdp:
        if rank == 0:
            print(f"--> using Lora for low cpu fsdp and low rank...")
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
            
        model=accelerator.prepare(model)
        
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
    # 这里应该不需要判断  rank==0
    
    train_loader = accelerator.prepare(train_loader)
    val_loader = accelerator.prepare(val_loader)
    accelerator.print(f"--> successfully prepared data loaders!")
    
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

    # fsdp activation checkpointing
  

    optimizer = get_optimizer(model, train_args)


    scheduler = get_scheduler(optimizer, train_args)
    optimizer,scheduler=accelerator.prepare(optimizer,scheduler)
    accelerator.print(f"--> successfully prepared optimizer and scheduler!")
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
            trial=trial,
            accelerator=accelerator,
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
            accelerator=accelerator
        )
        trial.report(validate_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    accelerator.print(f"starting saving model...........")
    model_state=accelerator.get_state_dict(model)
    model_save_path=os.path.join(train_args,nowtime+str(trial.number)+f'{train_args.epochs}.pth')
    accelerator.save(model_state,f=model_save_path)
    accelerator.print(f"saving the model successfully")
    return validate_loss


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
