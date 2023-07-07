import pickle
from collections import OrderedDict
import numpy as np

def save_pkl(file_path, data):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_pkl(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def check_pkl(file_path):
    data = load_pkl(file_path)
    if type(data) == OrderedDict:
        for k, v in data.items():
            print(k)
            for kk, vv in v.items():
                if type(vv) in [np.ndarray, np.array]:
                    print(f"\t{kk}: {type(vv)}")
                elif type(vv) == str:
                    print(f"\t{kk}: {vv}")
                elif type(vv) == tuple:
                    print(f"\t{kk}: {vv}")
                else:
                    print(f"\t{kk}: {type(vv)}")
    else:
        print(f"Expect pkl root data: collections.OrderedDict or Dict, got {type(data)}")


if __name__ == "__main__":
    def plus_one(num):
        return num + 1

    data = {
        'object': np.zeros((3, 3)),
        'fn': plus_one,
        'value': 3
    }

    save_pkl("tmp.pkl", data)
    newdata = load_pkl("tmp.pkl")

    print(newdata['object'])
    print(newdata['fn'](1))
    print(newdata['value'])

    binary_data = bytes("s", 'utf-8')

    for byte in binary_data:
        print(bin(byte)[2:].zfill(8), end=' ')

    check_pkl("diff.pkl")