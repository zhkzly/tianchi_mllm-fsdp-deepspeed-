import os

os.environ["HF_HOME"] = "huggingface"


from rich import print
import logging
import json
from tqdm import tqdm
import re
from src.utils.fms_fsdp.utils.dummy_data_utils import convert_to_json
from src.utils.fms_fsdp.utils.dummy_datasets import CustomSftDataset
from torch.utils.data import DataLoader
from typing import List, Dict, Any
import torch
import math

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(lineno)s",
)
logger = logging.getLogger(__name__)


from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


# outputs = deepcopy(self.data[index])
# # print(f"outputs:{outputs}")
# outputs["image"] = images
# outputs["image_id"] = self.data[index]["image"]
# # list[{},{},]
# outputs["instruction"] = self._help_genereate_prompt(outputs)
def collate_fn(batch: List[Dict[str, Any]]):
    inputs_text = [instance["instruction"] for instance in batch]
    inputs_image = [instance["image"] for instance in batch]
    # print(f"the type of inputs_image[0]:{type(inputs_image[0])}")
    data_ids = [instance["id"] for instance in batch]
    # 预测的时候没有label
    # labels=[instance['output'] for instance in batch]
    image_ids = [instance["image_id"] for instance in batch]
    # return inputs_text, inputs_image, data_ids, image_ids, labels
    return inputs_text, inputs_image, data_ids, image_ids


# 需要保存id，label，pre


def main(data_path="datas/train", task_type="0", data_type="train", device="cuda"):
    # Load the model in half-precision on the available device(s)
    device = torch.device(device)
    using_compile = True
    batch_size = 1
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map=device
    )
    if using_compile:
        print(f"compling the model")
        model = torch.compile(model)
    print(f"the device of model:{model.device}")
    # Qwen2VLForConditionalGeneration
    logger.info(f"the type of model:{type(model)}")
    print(model)
    # Qwen2VLProcessor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    logger.info(f"the type of processor:{type(processor)}")

    data_set = CustomSftDataset(
        preprocessor=processor,
        data_path=data_path,
        data_type=data_type,
        task_type=task_type,
    )
    total_data = len(data_set)
    print(f"total data:{total_data}")
    test_loader = DataLoader(
        dataset=data_set, batch_size=batch_size, collate_fn=collate_fn, num_workers=2
    )
    # print('+++++++++++++++++++++++++++')
    # print(f"the second batch:{data_set[1]}")
    # print('+++++++++++++++++++++++++++')
    datas = {"predict": [], "id": [], "image_id": [], "label": [], "origin_output": []}
    # datas = {"predict": [], "id": [], "image_id": [],'origin_output':[]}
    print("starting the data process")

    def resize_image(images, scale=2):
        return_images = []
        for image in images:

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
                new_w, new_h = math.floor(origin_w / 5), math.floor(origin_h / 5)
            elif origin_w < 500 or origin_h < 500:
                new_w, new_h = math.floor(origin_w / 1), math.floor(origin_h / 1)
            else:
                new_w, new_h = math.floor(origin_w / scale), math.floor(
                    origin_h / scale
                )
            logger.debug(f"the origin_w,origin_h:{origin_w,origin_h}")

            image = image.resize((new_w, new_h))
            logger.debug(f"the new_w,new_h:{new_w,new_h}")
            return_images.append(image)
        return return_images

    # inputs_text,inputs_image,data_ids,labels,image_ids
    model.eval()
    for inputs_text, inputs_image, data_ids, image_ids, labels in tqdm(
        test_loader, total=len(test_loader), colour="blue"
    ):
        inputs_image = [resize_image(image) for image in inputs_image]
        # print(f"++++++++++++++++++++++++++++++++")
        # print(f"the inputs_text:{inputs_text[0]}")
        # print(f"the inputs_image_id:{data_ids[0]}")
        print(f"the inputs image size:{inputs_image[0][0].size}")
        post_process_data = processor(
            text=inputs_text, images=inputs_image, padding=True, return_tensors="pt"
        )
        post_process_data = post_process_data.to(device)
        post_process_data = post_process_data.to(device)
        # print(f"the data_ids:{image_ids}")
        with torch.no_grad():
            output = model.generate(
                **post_process_data,
                max_new_tokens=128,
            )
            #     print(f"++++++++++++++++++++++++++++++++")
            #     output = processor.batch_decode(
            #     output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            # )
            #     print(f"the output:{output}")
            #     print(f"++++++++++++++++++++++++++++++++")

            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(post_process_data.input_ids, output)
            ]

            output_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            # print(f"the output_text:{output_text}")
            # break
            datas["origin_output"] += output_text
            datas["predict"] += [convert_to_json(text) for text in output_text]
            datas["id"] += data_ids
            datas["label"] += labels
            # [[],[]]
            datas["image_id"] += image_ids
    # for key ,value in datas.items():
    #     print(f"{key}:{len(value)}")
    #     print(f"{key}:{value[:5]}")
    data_save_path = os.path.join(data_path, f"task_{task_type}_{data_type}_pred.csv")
    import pandas as pd

    df = pd.DataFrame(datas)
    df.to_csv(data_save_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    data_path = "./datas/train"
    task_type = "1"
    data_type = "val"

    for task_type in ["all"]:
        main(data_path=data_path, task_type=task_type, data_type=data_type)
