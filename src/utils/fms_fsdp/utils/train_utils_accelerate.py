import os
from dataclasses import asdict
from functools import partial


try:
    import packaging.version
except ImportError:
    from pkg_resources import packaging  # type: ignore

import time
from datetime import timedelta

import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import ShardingStrategy

from src.utils.fms_fsdp.policies import *
import wandb
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
import datetime
MODEL_OUTPUT_CONFIG = {
    "use_cache": False,
    "return_dict": True,
    "output_hidden_states": False,
    "output_attentions": False,
    "output_attentions": False,
}


def merge_dict(dict1, dict2):
    output = {**dict1, **dict2}
    return output

from accelerate import Accelerator
def train(
    train_args,
    model,
    local_rank,
    rank,
    train_loader,
    optimizer,
    scheduler,
    profiler,
    start_step=0,
    trial=None,
    accelerator:Accelerator=None,
):
    # TODO:这个地方的记录应该是需要考虑在哪里声明

    dist.barrier()
    model.train()
    ddp_stats = torch.zeros(3).to(local_rank)

    start = time.time()
    loop_start = time.time()
    train_loss = -1
    # 混合精度
    optimizer.zero_grad()
    # 指定了一个开始的节点
    for batch_idx, (input, label) in enumerate(train_loader, start=start_step + 1):
        if batch_idx > train_args.max_steps:
            break
        if rank == 0:
            print(f"the shape of input is {input['input_ids'].shape}")
            print(f"the shape of label is {label.shape}")
        input = input.to(local_rank)
        label = label.to(local_rank)
        input_dict = merge_dict(MODEL_OUTPUT_CONFIG, input)
        with accelerator.accumulate(model):            
            with accelerator.autocast():
                output= model(labels=label, **input_dict)
            loss = (
                output.loss
                if hasattr(output, "loss")
                else output
            )
            accelerator.backward(loss)
            # 这里应该是有问题的
            if accelerator.sync_gradients:
                ddp_stats[1]+=accelerator.clip_grad_norm_(model.parameters(), train_args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        ddp_stats[0] += loss.item()
        
        # TODO: 这里需要修改成为batch_size
        ddp_stats[2] += 1

        if profiler:
            profiler.step()

        if batch_idx % train_args.report_interval == 0:
            # 等价与 all_gather+ reduceOp.SUM,all_gather,是得到所有的结果，
            # dist.all_reduce(ddp_stats, op=dist.ReduceOp.SUM)
            accelerator.reduce(ddp_stats,reduction='sum')
            train_loss = train_args.accumulation_steps * ddp_stats[0] / ddp_stats[2]
            g_norm = ddp_stats[1] / ddp_stats[2]
            elapsed_time = time.time() - loop_start
            # # 实际上不是token,因为每个的sequence 长度无法确定
            # new_tokens_seen = (
            #     (batch_idx - start_step) * world_size * train_args.batch_size
            # )
            if rank == 0:
                current_loss = train_loss.item()
                current_lr = scheduler.get_last_lr()[0]
                current_gnorm = g_norm.item()

                reserved_mem = torch.cuda.max_memory_reserved(
                    device=torch.cuda.current_device()
                )
                allocated_mem = torch.cuda.max_memory_allocated(
                    device=torch.cuda.current_device()
                )
                # trial.report(current_loss, batch_idx)

                print("step:", batch_idx)
                print("loss:", current_loss)
                print("LR:", current_lr)
                # print("tokens seen:", total_tokens_seen)
                print("gradient norm:", current_gnorm)
                print("reserved memory:", reserved_mem)
                print("allocated memory:", allocated_mem)
                print("one epoch time:", time.time() - start)

                if train_args.tracker:
                    vals_to_track = {
                        "learning rate": current_lr,
                        "loss": current_loss,
                        "gradient norm": current_gnorm,
                        "gpu reserved memory": reserved_mem,
                        "gpu allocated memory": allocated_mem,
                    }
            start = time.time()
            ddp_stats.zero_()
        torch.cuda.reset_peak_memory_stats(device=torch.cuda.current_device())

        if (batch_idx + 1) % train_args.checkpoint_interval == 0:
            nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            model_state=accelerator.get_state_dict(model)
            model_save_path=os.path.join(train_args,nowtime+str(trial.number)+f'{batch_idx}.pth')
            accelerator.save(model_state,f=model_save_path)
    return vals_to_track


def validate_fn(model, val_loader, epoch, train_args, rank, local_rank, tracker_fn,accelerator:Accelerator=None):
    if rank == 0:
        print(f"Validating at epoch {epoch}...")
    autocast = torch.amp.autocast(
        device_type="cuda", enabled=train_args.mixed_precision
    )

    with torch.no_grad():
        model.eval()
        val_stats = torch.zeros(2).to(local_rank)
        for val_batch_idx, (val_input, val_label) in enumerate(val_loader):
            val_input = val_input.to(local_rank)
            val_label = val_label.to(local_rank)
            val_input_dict = merge_dict(MODEL_OUTPUT_CONFIG, val_input)
            # with autocast:
            val_output = model(labels=val_label, **val_input_dict)
            val_loss = val_output.loss if hasattr(val_output, "loss") else val_output
            val_stats[0] += val_loss.item()
            val_stats[1] += val_input.shape[0]
        accelerator.reduce(val_stats)
        if rank == 0:
            mean_val_loss = val_stats[0] / val_stats[1]
            print(f"Validation loss at epoch {epoch}: {val_loss.item()}")
            if train_args.tracker:
                tracker_fn({"mean_validation loss": mean_val_loss.item()}, step=epoch)

        return mean_val_loss.item()


def setup():
    dist.init_process_group("nccl", timeout=timedelta(seconds=60 * 60))


def setup_environ_flags():
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)


def get_mixed_precision_policy(train_args, rank):
    verify_bfloat_support = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

    if train_args.mixed_precision:
        bf16_ready = verify_bfloat_support
        if bf16_ready:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        else:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
    else:
        mixed_precision_policy = None
        if rank == 0:
            print(f"without mixed precision")

    return mixed_precision_policy


def get_policies(train_args, rank, block=None):
    """Get policies for mixed precision, wrapping, sharding, ac and param init function."""

    # mixed precision
    mixed_precision_policy = get_mixed_precision_policy(train_args, rank)

    # wrapping policy
    wrapping_policy = get_wrapper(train_args, block)

    # sharding strategy
    if train_args.sharding_strategy == "fsdp":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif train_args.sharding_strategy == "hsdp":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif train_args.sharding_strategy == "ddp":
        sharding_strategy = ShardingStrategy.NO_SHARD
    else:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    if rank == 0:
        print(f"Sharding strategy = {train_args.sharding_strategy}")

    # ac handler
    # 不使用，否则需要配置一些block,param_init_function
    apply_selective_ac = partial(apply_fsdp_checkpointing, block=block)

    # param init function,避免在cpu上采用过多的内存，需要根据模型修改
    if train_args.low_cpu_fsdp:
        param_init_fn = param_init_function
    else:
        param_init_fn = None

    return (
        mixed_precision_policy,
        wrapping_policy,
        sharding_strategy,
        apply_selective_ac,
        param_init_fn,
    )


def get_profiler(train_args, rank):
    if not train_args.use_profiler:
        return
    if train_args.profiler_rank0_only and rank != 0:
        return
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            train_args.profile_traces
        ),
        profile_memory=True,
        with_stack=False,
        record_shapes=True,
    )


# 太多了
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLSdpaAttention,
    VisionAttention,
    VisionMlp,
    Qwen2MLP,
)
from torch.nn import Embedding

WRAP_BLOCKS = (Qwen2VLSdpaAttention, VisionAttention, VisionMlp, Qwen2MLP, Embedding)


def get_wrap_block(train_args, rank=0):
    if train_args.wrap_block is None:
        return None
    # set_block:List[str]
    set_block = WRAP_BLOCKS
    if rank == 0:
        print(f"set_block = {set_block}")
    return set_block


def get_tracker(train_args, rank=0):
    if rank == 0:
        print(f"--> tracker is enabling!")
        if train_args.tracker:
            if train_args.tracker not in ["wandb", "aim"]:
                raise ValueError(f"tracker {train_args.tracker} not supported.")
            tracker_dir = train_args.tracker_dir
            project_name = train_args.tracker_project_name
            run_id = train_args.tracker_run_id

            if train_args.tracker == "wandb":
                try:
                    import wandb  # type: ignore
                except ImportError:
                    raise ImportError(
                        "tracker is set to wandb but wandb is not installed."
                    )

                print(f"--> wandb is enabled!")
                try:
                    wandb.init(
                        project=project_name,
                        dir=tracker_dir,
                        resume="allow",
                        id=run_id,
                    )
                except wandb.errors.UsageError:
                    raise ValueError(
                        "wandb failed to init, did you pass your wandb api key via WANDB_API_KEY?"
                    )
                wandb.config = asdict(train_args)
                tracker_fn = wandb.log

            if train_args.tracker == "aim":
                try:
                    from aim import Run  # type: ignore
                except ImportError:
                    raise ImportError("tracker is set to aim but aim is not installed.")

                print(f"--> aim is enabled!")
                run = Run(
                    experiment=project_name,
                    repo=tracker_dir,
                    run_hash=run_id,
                )
                run["hparams"] = asdict(train_args)
                tracker_fn = run.track

    else:
        tracker_fn = lambda *args, **kwargs: None

    return tracker_fn
