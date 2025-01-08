




torchrun --nproc_per_node=1 src/hyper_search/optuna_office_fsdp.py \
             --data_path ./datas/train \
             --data_type train \
             --task_type 0 \
             --model_name Qwen/Qwen2-VL-2B-Instruct \
             --cache_dir ./huggingface/hub \
             --n_trials 20 \
             --timeout 600 \
             --n_jobs 1 \
             --direction minimize \
             --study_name mllm_hyper_search \
             --search_lr_min 1e-6 \
             --search_lr_max 1e-5 \
             --epochs_max 10 \
             --storage sqlite:///./logs/hyper_search/mllm.db \
             --load_if_exists True \
             --optuna_seed 678 \
             --low_rank_max 16 \
             --low_rank_min 4 \
             --low_rank_step 4 \
             --search_lora_alpha_min 4 \
             --search_lora_alpha_max 16 \
             --search_lora_alpha_step 4 \
             --low_rank 8 \
             --target_modules q_proj v_proj \
             --peft_type lora \
             --lora_alpha 8 \
             --bias none \
             --adapter_name sft_qwen_vl \
             --mixed_precision True \
             --wrap_block transformer \
             --wrapper_type transformer \
             --min_num_params 1e6 \
             --sharding_strategy fsdp \
             --low_cpu_fsdp True \
             --fsdp_activation_checkpointing True \
             --selective_checkpointing 1/3 \
             --use_torch_compile True \
             --use_profiler True \
             --profiler_rank0_only True \
             --grad_clip_thresh 2.0 \
             --tracker wandb \
             --tracker_dir ./logs/tracker \
             --tracker_project_name mllm \
             --use_lora True \
             --shuffle True \
             --num_workers 2 \
             --ckpt_save_path ./checkpoints/ \
             --finetune True \
             --gradient_accumulation_steps 1 \
             --report_interval 100 \
             --checkpoint_interval 1000 \
             --save_only_rank0 False    \
             --device_map cpu












































