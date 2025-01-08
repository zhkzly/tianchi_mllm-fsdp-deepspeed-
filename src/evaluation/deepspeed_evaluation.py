



# 加载分片的模型参数，由于没有将封装的模型层进行重新命名，所以需要在fsdp 封装后进行加载
# 是否可以完全完全是用meta device

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist

import torch
import os

from peft import get_peft_model, LoraConfig

from src.utils.fms_fsdp.utils.dummy_dataloader import get_dataloaders
from src.utils.fms_fsdp.utils.dummy_datasets import CustomSftDataset

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from tqdm import tqdm
from src.utils.fms_fsdp.utils.dummy_data_utils import convert_to_json
from src.utils.fms_fsdp.utils.train_utils import (
    get_policies,
    get_profiler,
    setup_environ_flags,
    get_wrap_block,
)
from src.hyper_search.optuna_helper import Checkpointer
from src.evaluation.batch_evaluation import collate_fn
import math
from src.hyper_search.configs import FsdpEvaluationArgs
import deepspeed
from torch.utils.data import DataLoader
class FsdpEvaluation:
    def __init__(self, fsdp_evaluation_args: FsdpEvaluationArgs):
        self.evaluation_args = fsdp_evaluation_args
        # self.set_up()
        # self.rank = dist.get_rank()

    def set_up(self,rank,world_size):
        torch.cuda.empty_cache()
        setup_environ_flags()
        torch.backends.cudnn.enabled = False
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        dist.init_process_group(backend="nccl",rank=rank,world_size=world_size)
        torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())


    def evaluate_pretrain_model(self,rank,world_size):
        # copy optuna_helper.py
        self.set_up(rank=rank,world_size=world_size)
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        if self.evaluation_args.low_cpu_fsdp:
            print(f"--> using Lora for low cpu fsdp and low rank...")
            print("Loading model...")
            print(
                f"the loading config of model is {self.evaluation_args.model_name}"
            )
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.evaluation_args.model_name,
                cache_dir=self.evaluation_args.cache_dir,
                torch_dtype=torch.float32,
                device_map="cpu",
            )
            for param in model.parameters():
                param.requires_grad = False

        # get data loader
        print("Constructing datasets...")
        print(f"--> using data_path: {self.evaluation_args.data_path}")
        print(f"loading qwen2 processor...")
        print("Datasets constructed!")
        processor = AutoProcessor.from_pretrained(
            self.evaluation_args.model_name, cache_dir=self.evaluation_args.cache_dir
        )
        val_dataset = CustomSftDataset(
            preprocessor=processor,
            data_path=self.evaluation_args.data_path,
            data_type=self.evaluation_args.data_type,
            task_type=self.evaluation_args.task_type,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.evaluation_args.batch_size,
            num_workers=self.evaluation_args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        print("Datasets constructed!")
        print("successfully constructed data loaders!")

        # FSDP,应该是在这里进行了模型参数的初始化，也就是放置到了实际的gpu上

        print(f"--> initializing fsdp model...")

        
        model.eval()
        model=deepspeed.init_inference(model, mp_size=world_size, dtype=torch.float32, replace_method='auto')

        pbar = tqdm(
            val_loader,
            total=len(val_loader),
            colour="blue",
            desc=f"evaluation epochs",
            disable=(rank != 0),
        )

        datas = {"predict": [], "id": [], "image_id": [], "origin_output": []}
        # datas = {
        #     "predict": [],
        #     "id": [],
        #     "image_id": [],
        #     "label": [],
        #     "origin_output": [],
        #     "label": [],
        # }
        print("starting the data process")
        # generate_config={'return_'}
        for inputs_text, inputs_image, data_ids, image_ids in pbar:
            # for inputs_text, inputs_image, data_ids, image_ids, labels in pbar:
            inputs_image = [self._resize_image(image) for image in inputs_image]
            # print(f"++++++++++++++++++++++++++++++++")
            # print(f"the inputs_text:{inputs_text[0]}")
            # print(f"the inputs_image_id:{data_ids[0]}")
            # print(f"the inputs image size:{inputs_image[0][0].size}")
            post_process_data = processor(
                text=inputs_text, images=inputs_image, padding=True, return_tensors="pt"
            )
            post_process_data = post_process_data.to(torch.cuda.current_device())
            # print(f"the data_ids:{image_ids}")
            # post_process_data = post_process_data.to(torch.cuda.current_device())
            with torch.no_grad():
                output = model.generate(
                    **post_process_data,
                    max_new_tokens=128,
                )
                # print(f"the type of output:{type(output)}")
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(
                        post_process_data.input_ids, output
                    )
                ]
                # dist.gather(generated_ids, dst=0)

                output_text = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

                datas["origin_output"] += output_text
                datas["predict"] += [convert_to_json(text) for text in output_text]
                datas["id"] += data_ids
                # datas["label"] += labels
                # [[],[]]
                datas["image_id"] += image_ids

        data_save_path = os.path.join(
            self.evaluation_args.data_path,
            f"task_{self.evaluation_args.task_type}_{self.evaluation_args.data_type}_pred_rank_{rank}.csv",
        )
        print(f"saving the data to {data_save_path}")
        import pandas as pd

        df = pd.DataFrame(datas)
        df.to_csv(data_save_path, index=False, encoding="utf-8-sig")
        print(f"successfully saved the data to {data_save_path}")
        from src.evaluation.convert2submit import convert2submit, merge_csv

        if rank == 0:
            file_names = [
                f"task_{self.evaluation_args.task_type}_{self.evaluation_args.data_type}_pred_rank_{i}.csv"
                for i in range(world_size)
            ]
            merge_csv(
                self.evaluation_args.data_path,
                file_names=file_names,
                save_name="merged_pred.csv",
            )
            convert2submit(
                self.evaluation_args.data_path,
                file_name="merged_pred.csv",
                save_name="submit.csv",
            )
        print(f"successfully convert the data to submit format")
        dist.barrier()
        dist.destroy_process_group()
        return None

    def evaluate_sft_model(self):

        # copy optuna_helper.py

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = rank % torch.cuda.device_count()
        # get policy
        # 给哪个 module 进行封装，在这里采用尺寸进行判断，需要和训练的一致
        # such as transformer_block
        block = get_wrap_block(self.evaluation_args, rank)
        (
            mixed_precision_policy,
            wrapping_policy,
            sharding_strategy_policy,
            apply_selective_ac,
            param_init_fn,
        ) = get_policies(self.evaluation_args, rank, block)

        # get fms model
        # llama_config = get_model_config(self.evaluation_args.model_name)
        if self.evaluation_args.low_cpu_fsdp:
            if rank == 0:
                print(f"--> using Lora for low cpu fsdp and low rank...")
                if self.evaluation_args.use_lora:
                    if self.rank == 0:
                        print("Loading model...")
                        print(
                            f"the loading config of model is {self.evaluation_args.model_name}"
                        )
                        model = Qwen2VLForConditionalGeneration.from_pretrained(
                            self.evaluation_args.model_name,
                            cache_dir=self.evaluation_args.cache_dir,
                            torch_dtype=torch.float32,
                            device_map="cpu",
                        )
                        lora_config = LoraConfig(
                            r=self.evaluation_args.low_rank,
                            target_modules=self.evaluation_args.target_modules,
                            peft_type=self.evaluation_args.peft_type,
                            lora_alpha=self.evaluation_args.lora_alpha,
                            bias=self.evaluation_args.bias,
                        )
                        model = get_peft_model(
                            model=model,
                            peft_config=lora_config,
                            adapter_name=self.evaluation_args.adapter_name,
                        )
                        param_init_fn = None
            else:
                with torch.device("meta"):
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        self.evaluation_args.model_name,
                        cache_dir=self.evaluation_args.cache_dir,
                        torch_dtype=torch.float32,
                    )
                    lora_config = LoraConfig(
                        r=self.evaluation_args.low_rank,
                        target_modules=self.evaluation_args.target_modules,
                        peft_type=self.evaluation_args.peft_type,
                        lora_alpha=self.evaluation_args.lora_alpha,
                        bias=self.evaluation_args.bias,
                    )
                    model = get_peft_model(
                        model=model,
                        peft_config=lora_config,
                        adapter_name=self.evaluation_args.adapter_name,
                    )

                def param_init_fn(module):
                    return module.to_empty(
                        device=torch.cuda.current_device(), recurse=False
                    )

        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n--> model has {total_params / 1e6} Million params\n")

        # get data loader
        if rank == 0:
            print("Constructing datasets...")
            print(f"--> using data_path: {self.evaluation_args.data_path}")
            print(f"loading qwen2 processor...")
            print("Datasets constructed!")
        processor = AutoProcessor.from_pretrained(self.evaluation_args.model_name)
        val_dataset = CustomSftDataset(
            preprocessor=processor,
            data_path=self.evaluation_args.data_path,
            data_type=self.evaluation_args.data_type,
            task_type=self.evaluation_args.task_type,
        )
        val_loader = get_dataloaders(
            None,
            val_dataset,
            world_size=world_size,
            local_rank=rank,
            shuffle=self.evaluation_args.shuffle,
            seed=self.evaluation_args.seed,
            collator=collate_fn,
            batch_size=self.evaluation_args.batch_size,
            num_workers=self.evaluation_args.num_workers,
        )

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
            # 必须和compile同时使用，否则会报错
            use_orig_params=self.evaluation_args.use_torch_compile,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            param_init_fn=param_init_fn,
            sync_module_states=True,
        )
        # we need this post-fsdp call to avoid graph break with torch.compile, until we figure out a better solution.
        # model.rot_emb.compute_freqs_cis(
        #     torch.device("cuda", torch.cuda.current_device()),
        #     model.config.max_expected_seq_len,
        # )

        # fsdp activation checkpointing
        if self.evaluation_args.fsdp_activation_checkpointing:
            if rank == 0:
                print(f"--> applying FSDP activation checkpointing...")
            apply_selective_ac(model, p=self.evaluation_args.selective_checkpointing)

        # torch compile
        if self.evaluation_args.use_torch_compile:
            if rank == 0:
                print(f"--> enabling torch compile...")
            # the default accumulated_cache_size_limit=64 is not enough for 70b model, so we make it 128 here
            torch._dynamo.config.accumulated_cache_size_limit = 128
            model = torch.compile(model)

        # optionally load from checkpoint (when continue pretraining)
        checkpointer = Checkpointer(
            self.evaluation_args.ckpt_save_path,
            self.evaluation_args.sharding_strategy,
            rank,
            local_rank,
        )

        checkpointer.load(
            model=model, optimizer=None, file_name=self.evaluation_args.load_file_name
        )

        # profiler
        profiler = get_profiler(self.evaluation_args, rank)

        pbar = tqdm(
            val_loader,
            total=len(val_loader),
            colour="blue",
            desc=f"evaluation epochs",
            disable=(rank != 0),
        )

        if rank == 0:
            # datas = {"predict": [], "id": [], "image_id": [],'origin_output':[]}
            datas = {
                "predict": [],
                "id": [],
                "image_id": [],
                "origin_output": [],
                "label": [],
            }
            print("starting the data process")
        for inputs_text, inputs_image, data_ids, image_ids, labels in pbar:
            inputs_image = [self._resize_image(image) for image in inputs_image]
            # print(f"++++++++++++++++++++++++++++++++")
            # print(f"the inputs_text:{inputs_text[0]}")
            # print(f"the inputs_image_id:{data_ids[0]}")
            # print(f"the inputs image size:{inputs_image[0][0].size}")
            post_process_data = processor(
                text=inputs_text, images=inputs_image, padding=True, return_tensors="pt"
            )
            post_process_data = post_process_data.to(torch.cuda.current_device())
            # print(f"the data_ids:{image_ids}")
            with torch.no_grad():
                output = model.generate(
                    **post_process_data,
                    max_new_tokens=128,
                )

                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(
                        post_process_data.input_ids, output
                    )
                ]

                output_text = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                # print(f"the output_text:{output_text}")
                # break
                if rank == 0:
                    datas["origin_output"] += output_text
                    datas["predict"] += [convert_to_json(text) for text in output_text]
                    datas["id"] += data_ids
                    # 产生submit 的时候应注释掉下面一行
                    datas["label"] += labels
                if profiler is not None:
                    profiler.step()
                # datas['label']+=labels
                # [[],[]]
                datas["image_id"] += image_ids
        # for key ,value in datas.items():
        #     print(f"{key}:{len(value)}")
        #     print(f"{key}:{value[:5]}")
        data_save_path = os.path.join(
            self.evaluation_args.data_path,
            f"task_{self.evaluation_args.task_type}_{self.evaluation_args.data_type}_pred.csv",
        )
        if rank == 0:
            print(f"saving the data to {data_save_path}")
            import pandas as pd

            df = pd.DataFrame(datas)
            df.to_csv(data_save_path, index=False, encoding="utf-8-sig")
            print(f"successfully saved the data to {data_save_path}")
            from src.evaluation.convert2submit import convert2submit

            convert2submit(
                self.evaluation_args.data_path,
                file_name=os.path.basename(data_save_path),
            )
            print(f"successfully convert the data to submit format")
        dist.destroy_process_group()
        return None

    @staticmethod
    def _resize_image(images, scale=2):
        return_images = []
        for image in images:
            # print(f"the type of image:{type(image)}")
            origin_w, origin_h = image.size
            if origin_w > 5000 or origin_h > 5000:
                new_w, new_h = math.floor(origin_w / 10), math.floor(origin_h / 10)
            elif origin_w > 4000 or origin_h > 4000:
                new_w, new_h = math.floor(origin_w / 9), math.floor(origin_h / 9)
            elif origin_w > 3000 or origin_h > 3000:
                new_w, new_h = math.floor(origin_w / 8), math.floor(origin_h / 8)
            elif origin_w > 2000 or origin_h > 2000:
                new_w, new_h = math.floor(origin_w / 6), math.floor(origin_h / 6)
            elif origin_w > 1000 or origin_h > 1000:
                new_w, new_h = math.floor(origin_w / 4), math.floor(origin_h / 4)
            elif origin_w < 500 or origin_h < 500:
                new_w, new_h = math.floor(origin_w / 1), math.floor(origin_h / 1)
            else:
                new_w, new_h = math.floor(origin_w / scale), math.floor(
                    origin_h / scale
                )
            # logger.debug(f"the origin_w,origin_h:{origin_w,origin_h}")

            image = image.resize((new_w, new_h))
            # logger.debug(f"the new_w,new_h:{new_w,new_h}")
            return_images.append(image)
        return return_images


from transformers import HfArgumentParser
import torch.multiprocessing as mp
if __name__ == "__main__":
    eval_parser = HfArgumentParser(FsdpEvaluationArgs)
    evaluation_args = eval_parser.parse_args_into_dataclasses()[0]
    evaluater = FsdpEvaluation(evaluation_args)
    if evaluation_args.sft:
        evaluater.evaluate_sft_model()
    else:
        world_size = torch.cuda.device_count()
        mp.spawn(evaluater.evaluate_pretrain_model, args=(world_size,), nprocs=world_size)




























