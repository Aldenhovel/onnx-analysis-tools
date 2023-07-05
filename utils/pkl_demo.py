import collections
import pickle
from collections import OrderedDict
import numpy as np


def save_pkl(file_path: str, data: collections.OrderedDict):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_pkl(file_path: str) -> collections.OrderedDict:
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def check_pkl(file_path: str):
    data = load_pkl(file_path)
    if type(data) == OrderedDict:
        for k, v in data.items():
            print(k)
            for kk, vv in v.items():
                if type(vv) in [np.ndarray, np.array]:
                    print(f"\t{kk}: {type(vv)} ,shape is {vv.shape}")
                elif type(vv) == str:
                    print(f"\t{kk}: {vv}")
                else:
                    print(f"\t{kk}: {type(vv)}")


if __name__ == "__main__":

    print("=" * 20 + "CHECK THE PKL FILE" + "=" * 20)
    # 概览 pkl 内容
    check_pkl("diff.pkl")

    # 加载 pkl ,这里 diff.pkl 序列化之前是个 OrderedDict 加载后也是 OrderedDict
    data = load_pkl("diff.pkl")
    assert type(data) == collections.OrderedDict

    print("\n\n\n")
    print("=" * 20 + "LOAD RECORDS" + "=" * 20)
    # 读取第一条记录（第i条记录对应模型从input到第i层作为输出的结果，例如第一条即仅包含开头的一个层，第二条即包含开头的两个层，以此类推）
    print(f"output name: {data[0]['name']}")
    print(f"onnxruntime: {data[0]['ort']}")
    print(f"r8: {data[0]['r8']}")
    print(f"diff: {data[0]['diff']}")

    print("\n\n\n")
    print("=" * 20 + "SEARCH RECORDS" + "=" * 20)
    # 查找从input到某一层的子图
    output_name = "/layer4/layer4.1/Add_output_0"
    for ix, record in data.items():
        if record['name'] == output_name:
            print(record)
