from dataclasses import dataclass
from typing import Dict, Sequence, List
from torch.utils.data import Dataset
from copy import deepcopy
import json
import torch
import transformers
from src.utils.fms_fsdp.utils.dummy_data_utils import Conversions
import os
from PIL import Image

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<|pad|>"
DEFAULT_EOS_TOKEN = "<|im_end|>"
DEFAULT_UNK_TOKEN = "<|unk|>"


# https://github.com/abacaj/train-with-fsdp
# PROMPT_DICT = {
#     "instruct_prompt": "[Instructions]:\n{instruction}\n\n[Response]:",
# }


# def _tokenize_fn(
#     strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
# ) -> Dict:
#     """Tokenize a list of strings."""
#     tokenized_list = [
#         tokenizer(
#             text,
#             return_tensors="pt",
#             padding="longest",
#             max_length=tokenizer.model_max_length,
#             truncation=True,
#         )
#         for text in strings
#     ]

#     input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
#     input_ids_lens = labels_lens = [
#         tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
#         for tokenized in tokenized_list
#     ]
#     return dict(
#         input_ids=input_ids,
#         labels=labels,
#         input_ids_lens=input_ids_lens,
#         labels_lens=labels_lens,
#     )


# def preprocess(
#     sources: Sequence[str],
#     targets: Sequence[str],
#     tokenizer: transformers.PreTrainedTokenizer,
# ) -> Dict:
#     """Preprocess the data by tokenizing."""
#     examples = [s + t for s, t in zip(sources, targets)]
#     examples_tokenized, sources_tokenized = [
#         _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
#     ]
#     input_ids = examples_tokenized["input_ids"]
#     labels = copy.deepcopy(input_ids)
#     for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
#         label[:source_len] = IGNORE_INDEX

#     return dict(input_ids=input_ids, labels=labels)


# def load_data(files):
#     data = []

#     for file in files:
#         f = open(file, mode="r")
#         if "jsonl" in file:
#             _data = f.read()
#             _data = _data.splitlines()
#             _data = [json.loads(js) for js in _data]
#         else:
#             _data = json.load(f)

#         f.close()

#         data += _data

#     return data


# def process_example(example: Dict):
#     return example.get("output")


# def format_sources(example):
#     return PROMPT_DICT["instruct_prompt"].format_map(example)


# class SupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""

#     def __init__(
#         self,
#         tokenizer: transformers.PreTrainedTokenizer,
#         data_path: list[str],
#     ):
#         super(SupervisedDataset, self).__init__()
#         list_data_dict = load_data(data_path)
#         sources = [format_sources(example) for example in list_data_dict]
#         targets = [
#             f"{process_example(example)}{tokenizer.eos_token}"
#             for example in list_data_dict
#         ]

#         print("Tokenizing inputs... This may take some time...")
#         data_dict = preprocess(sources, targets, tokenizer)

#         self.input_ids = data_dict["input_ids"]
#         self.labels = data_dict["labels"]

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         return dict(input_ids=self.input_ids[i], labels=self.labels[i])


# @dataclass
# class DataCollatorForSupervisedDataset(object):
#     """Collate examples for supervised fine-tuning."""

#     tokenizer: transformers.PreTrainedTokenizer

#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         input_ids, labels = tuple(
#             [instance[key] for instance in instances] for key in ("input_ids", "labels")
#         )
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#         )
#         labels = torch.nn.utils.rnn.pad_sequence(
#             labels, batch_first=True, padding_value=IGNORE_INDEX
#         )

#         return dict(
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#         )


class CustomSftDataset(Dataset):
    #
    def __init__(
        self,
        preprocessor,
        data_path,
        data_type="train",
        task_type="all",
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.data_path = data_path
        self.data_type = data_type
        self.task_type = task_type
        self.conversion = Conversions()
        self._load_data()

    def _load_data(self):
        data_path = os.path.join(self.data_path, self.data_type + ".json")
        with open(data_path, "r") as f:
            self.data = json.load(f)[self.task_type]
            if self.data is None:
                print(f"loading {self.data_type} data failed...")

    def _help_genereate_prompt(self, instruction):
        instruction = self.conversion.parse_conversation(instruction["instruction"])
        # add_general_prompt=True  好像没用add_generation_prompt
        instruction = self.preprocessor.apply_chat_template(instruction, add_generation_prompt=True)
        # print(f"batch_size_evaluate instruction:{instruction}")
        return instruction

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        images = [
            Image.open(os.path.join(self.data_path, "images", image))
            for image in self.data[index]["image"]
        ]
        outputs = deepcopy(self.data[index])
        # print(f"outputs:{outputs}")
        outputs["image"] = images
        outputs["image_id"] = self.data[index]["image"]
        # list[{},{},]
        outputs["instruction"] = self._help_genereate_prompt(outputs)
        return outputs


# def collate_fn(batch:List[List[Dict]]):
#     # helping organize the return of the dataset call
#     # 需要进行批量化的padding，得到一些掩码，和ignore,构造完整的 sft instruction
import math

def resize_image(images,scale=3):
    tempt_images=[]
    if isinstance(images,list):
        if isinstance(images[0],list):
            for image_list in images:
                for image in image_list:
                    tempt_images.append(image)
    return_images=[]
    for image in tempt_images:
        origin_w, origin_h = image.size
        if origin_w > 5000 or origin_h >5000:
            new_w, new_h = math.floor(origin_w / 25), math.floor(origin_h / 25)
        elif origin_w > 4000 or origin_h >4000:
            new_w, new_h = math.floor(origin_w / 20), math.floor(origin_h / 20)
        elif origin_w > 3000 or origin_h >3000:
            new_w, new_h = math.floor(origin_w / 18), math.floor(origin_h / 18)
        elif origin_w > 2000 or origin_h >2000:
            new_w, new_h = math.floor(origin_w / 15), math.floor(origin_h / 15)
        elif origin_w > 1000 or origin_h >1000:
            new_w, new_h = math.floor(origin_w / 10), math.floor(origin_h / 10)
        elif origin_w > 500 or origin_h >500:
            new_w, new_h = math.floor(origin_w / 8), math.floor(origin_h / 8)
        else:
            new_w, new_h = math.floor(origin_w / scale), math.floor(origin_h / scale)
        # logger.debug(f"the origin_w,origin_h:{origin_w,origin_h}")
        
        image = image.resize((new_w, new_h))
        # logger.debug(f"the new_w,new_h:{new_w,new_h}")
        return_images.append(image)
    return return_images

class DataSftCollator:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.ignore_index = IGNORE_INDEX
        self.split_idx = self.preprocessor.tokenizer.convert_tokens_to_ids("\\n")
        self.pre_processor_kwargs = {"text_kwargs": {"padding": True}}

    def _get_full_prompts(self, instances):
        full_prompts = [
            instance["instruction"] + instance["output"] + DEFAULT_EOS_TOKEN
            for instance in instances
        ]
        return full_prompts

    def _get_exact_number_index(self, lst):
        for index, value in enumerate(reversed(lst)):
            if value == self.split_idx:
                # len(lst) - 1 - index 是为了计算原始列表中的索引位置
                last_index = len(lst) - 1 - index
                break
            else:
                last_index = None  # 如果没有找到，则设置为 None 或其他默认值

    def _get_response_start_idx(self, instances):
        response_index_list = [
            self._get_exact_number_index(instance)
            for instance in instances["input_ids"].tolist()
        ]
        return response_index_list

    def _get_labels(self, input_ids, response_index_list):
        labels = []
        input_length = input_ids.shape[1]
        for ignore_length, input_id in zip(response_index_list, input_ids):
            ignore_index = torch.zeros_like(input_ids[0], dtype=torch.long)
            ignore_index[:ignore_length] = self.ignore_index
            label = torch.where(
                ignore_index == self.ignore_index, ignore_index, input_id
            )
            labels.append(label)
        return torch.stack(labels, dim=0)

    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        # 将 response和instruction进行拼接获取一个完整的text list,获取一个完整的image list
        #  之后进行preprocessor ，得到模型的输入
        full_prompts = self._get_full_prompts(instances)
        full_prompts=[prompt[:20] for prompt in full_prompts]
        images = [instance["image"] for instance in instances]
        inputs_features_dict = self.preprocessor(
            text=full_prompts,
            images=images,
            return_tensors="pt",
            **self.pre_processor_kwargs,
        )
        # print("##############################")
        # print(f"the type of inputs_features_dict:{type(inputs_features_dict)}")
        # print(f"the keys of inputs_features_dict:{inputs_features_dict.keys()}")
        # print(f"the type of input_ids:{type(inputs_features_dict['input_ids'])}")
        # print(f"the shape of input_ids:{inputs_features_dict['input_ids'].shape}")
        # print(f"to_list of input_ids:{inputs_features_dict['input_ids'].tolist()}")
        # print("##############################")
        response_index_list = self._get_response_start_idx(inputs_features_dict)

        input_ids = inputs_features_dict["input_ids"]
        labels = self._get_labels(input_ids, response_index_list=response_index_list)
        # print(f"the shape of input_ids:{input_ids.shape}")
        # print(f"the shape of labels:{labels.shape}")
        return inputs_features_dict, labels



if __name__ == "__main__":
    from transformers import AutoProcessor, AutoTokenizer
    from dataclasses import dataclass

    @dataclass
    class ModelArgs:
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
        cache_dir = "./huggingface/hub"

    model_args = ModelArgs()
    preprocessor = AutoProcessor.from_pretrained(model_args.model_name)
    train_dataset = CustomSftDataset(
        preprocessor=preprocessor,
        data_path="./datas/train",
        data_type="val",
        task_type="all",
    )
    data_collator = DataSftCollator(preprocessor=preprocessor)
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        collate_fn=data_collator,
    )

