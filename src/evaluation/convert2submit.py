import json
from pathlib import Path
import pandas as pd


# def convert2submit(test_file: Path, prediction_file: Path, save_path: Path):
#     pred_label_list = []

#     for line in open(prediction_file, "r"):
#         prediction_data = json.loads(line)

#         pred_label = prediction_data["predict"]
#         pred_label_list.append(pred_label)

#     test_data = json.load(open(test_file, "r"))
#     save_data = []
#     for i, example in enumerate(test_data):
#         example["predict"] = pred_label_list[i]
#         save_data.append(example)

#     df = pd.DataFrame(save_data)

#     df.to_csv(save_path, index=None, encoding="utf-8-sig")

import pandas as pd
import os
from src.utils.fms_fsdp.utils.dummy_data_utils import convert_to_json
import json


def convert_help(item):
    # print(f"the type of item:{type(item)}")
    item = convert_to_json(item)

    return item[0]


def convert2submit(
    pred_file_path,
    file_name="task_all_train_pred.csv",
    columns_to_read=["id", "predict"],
    save_name="submit.csv",
):
    file_path = os.path.join(pred_file_path, file_name)
    df = pd.read_csv(filepath_or_buffer=file_path, usecols=columns_to_read)
    df["predict"] = df["predict"].apply(convert_help)
    print(df)
    save_path = os.path.join(pred_file_path, save_name)
    df.to_csv(save_path, index=None, encoding="utf-8-sig")
    print("saving successfully !")


def merge_csv(
    pred_file_path,
    file_names=["task_all_train_pred_rank0.csv", "task_all_train_pred_rank0.csv"],
    save_name="merged_cleaned_file.csv",
):
    # 读取两个 CSV 文件
    df_list = [
        pd.read_csv(os.path.join(pred_file_path, file_name)) for file_name in file_names
    ]

    # 将两个 DataFrame 合并在一起
    combined_df = pd.concat(df_list, ignore_index=True)

    # 根据指定列（例如 'id' 列）去除重复行
    # keep='first' 表示保留第一次出现的行，你可以选择 'last' 或 False 来改变行为
    cleaned_df = combined_df.drop_duplicates(subset=["id"], keep="first")

    # 如果需要，可以将结果保存到新的 CSV 文件中
    save_path = os.path.join(pred_file_path, save_name)
    cleaned_df.to_csv(save_path, index=False)

    print(f"Merged and cleaned file saved as {save_name}'")


if __name__ == "__main__":

    # pred_file_path = "../datas/test1"
    # convert2submit(pred_file_path=pred_file_path)
    data_path = "./datas/train"
    world_size = 2
    file_names = [f"task_0_test_pred_rank_{i}.csv" for i in range(world_size)]
    merge_csv(
        data_path,
        file_names=file_names,
        save_name="merged_pred.csv",
    )
    convert2submit(
        data_path,
        file_name="merged_pred.csv",
        save_name="submit.csv",
    )
# end main
