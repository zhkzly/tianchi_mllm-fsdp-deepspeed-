# note

```text
    the code may be not complete, and some code may be wrong,
    i will update it when i have time.

    but this can be used as a reference for you to understand the fsdp and optuna.


```

# some explanation

```text
# tianchi 多模态
 qwen2vl-2b
```

```text
try to sft the qwen2vl-2b,using fsdp
and using the optuna to find the best hyperparameters

in each file,the .sh file is the command to run the code,
and the .py file is the code. you can read the code from the .sh file,
and find the main function to run the code.

./src/exmple.ipynb:
    this is using for check the code (i may forget some function call,so i use this
    file to check the code),the code in this file is not ordered.i add some code cell randomly,


```

```text
./src/evaluation

some code for evaluation(fsdp evaluation,deepspeed evaluation)
for fsdp evaluation,there are some error .
firstly,fsdp  is not support the non forward function.,such model.generate(),this function will not activate the sync
https://github.com/pytorch/pytorch/issues/123962


secondly,so i use the deepspeed evaluation,

```

```text
./src/hyper_search
    i have use the optuna to find the best hyperparameters,
there are some dummy code for hyper_search(which can not be run),
```

```text
optuna_office_fsdp.py:
    using for fsdp,which is copy from the example by optuna official. i did not debug it yet.(don't have the environment to run it)
https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_distributed_simple.py
```

```text
the optuna_custom_fsdp.py:
    using for fsdp,which is my custom code for fsdp (which using the custom broadcast function to sync the config).  i did not debug it yet.

```

```text
the optuna_accelerate.py:
    using the accelerate to accelerate the training,which is a wrapper for fsdp

```

```text
there are some tricks for fsdp:
loading the model in only in rank 0(on cpu), and others use the meta device to load the model.
in fsdp config, we need set the param_init_fn,and sync_module_states to True,and set device_id=f'cuda:{dist.get_rank()}'
https://huggingface.co/blog/zh/pytorch-fsdp
https://zhuanlan.zhihu.com/p/671698810


when using the fsdp ,we may be meet some error,such as:
RuntimeError: The tensor has a non-zero number of elements, but its data is not allocated yet. Caffe2 uses a lazy allocation,
so you will need to call mutable_data() or raw_mutable_data() to actually allocate memory.
(https://github.com/pytorch/pytorch/issues/123962)

RuntimeError: Expected 3-dimensional input for 3-dimensional weight [64, 512, 1], but got 2-dimensional
 this often happen when the fsdp config is not correct.such the sync is not correct.

```

```text
other problems:
fsdp is not support the huggingface peft(lora) well
when the model param is not the same property(require_grad,dtype,device,need to be the same),it will cause the error.
ValueError: FlatParameter requires uniform requires_grad(https://github.com/pytorch/pytorch/issues/104690)
we need set the use_orig_param to True

```

```text
./src/search_hyper/ray_fsdp.py
    i have use the ray to search the hyperparameters,which is not complete.
    but i think it can be used as a reference for you.ray is much more simple than optuna.

```

```text
there are some other codes for fsdp:
[submodule "submodules/fms-fsdp"]
	path = submodules/fms-fsdp
	url = https://github.com/foundation-model-stack/fms-fsdp.git
[submodule "submodules/train-with-fsdp"]
	path = submodules/train-with-fsdp
	url = https://github.com/abacaj/train-with-fsdp.git

```

# conclusion

```text
 fsdp is not support the huggingface peft(lora) well,and the inference is not
suport the fsdp well(can not using the model.generate()).

maybe we can use the deepspeed to accelerate the training,
and the simple way is using the  huggingface "accelerate" which is support the
config of fsdp and deepspeed

for hyperparameters search,maybe we can use the optuna or ray.
and the ray is much powerful than optuna.


```
