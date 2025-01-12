
from transformers import AutoModelForCausalLM,AutoTokenizer
import os

def get_model_tokenizer(model_args,**kwargs):
    model_name=model_args.model_name
    cache_dir=model_args.cache_dir
    
    model_params_path=os.path.join(cache_dir,'model_params')
    tokenizer_path=os.path.join(cache_dir,'tokenizer')
    if not os.path.exists(model_params_path):
        os.makedirs(model_params_path)
        os.makedirs(tokenizer_path)

    # model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",cache_dir='./huggingface'
    # 完全采用了AutoModelCausalLM，可以采用 base model 也就是  QWenModel()
    tokenizer = AutoTokenizer.from_pretrained(model_name,torch_dtype="auto",
    device_map="auto",cache_dir=tokenizer_path,**kwargs)
    model=AutoModelForCausalLM.from_pretrained(model_name,cache_dir=model_params_path)

    return model,tokenizer  

































