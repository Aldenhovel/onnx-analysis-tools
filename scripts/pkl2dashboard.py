import os.path
import argparse
import sys
import pathlib
import os
import pandas as pd

sys.path.append(str(pathlib.Path(".").absolute().parent))

option_template = \
    """ 
    option = {
        grid: { left: '15%', right: '15%', top: '10%', bottom: '10%'},
        title: {text: "onnxruntime vs rpprt_r8 diff", subtext: 'from input to layer output abs error', left: 'center'},
        xAxis: { type: 'category', data: $layer_names$, show: false, name: 'Layers'},
        yAxis: { type: 'value', name: 'Mean ABS Error' },
        toolbox: { show: true, feature: { magicType: { show: true, type: ['line', 'bar'] }, restore: { show: true }, saveAsImage: { show: true }}},
        tooltip: { trigger: 'axis', position: function (pt) { return [pt[0], '10%']; }},
        dataZoom: [{ type: 'inside', start: 0, end: 100 }, { start: 0, end: 100 }, {show: true, yAxisIndex: 0, filterMode: 'empty', width: 30, height: '80%', showDataShadow: false, left: '86%' }],
        series: [{data: $diff_values$, type: 'bar' }]
    };
    """

import numpy as np
from utils.Pickle import load_pkl, check_pkl
from utils.StyleText import *


def main(pkl_file: str = "diff.pkl"):
    if not os.path.exists(pkl_file):
        print(f"{style_error()} pkl file not found.")

    check_pkl(pkl_file)

    data = load_pkl(pkl_file)
    check_pkl(pkl_file)

    layer_names, diff_values = [], []
    for layer_ix, v_dict in data.items():
        layer_names.append(f"{layer_ix}# {v_dict['name']}")
        diff_values.append(np.mean(v_dict['diff']))
    print(str(layer_names))
    print(str(diff_values))

    option = option_template.replace("$layer_names$", str(layer_names))
    option = option.replace("$diff_values$", str(diff_values))
    print(option)

    with open('../utils/dashboard.html', 'r', encoding='utf-8') as f:
        html_template = f.read()
    html = html_template.replace("$option$", option)
    print(html)

    with open('../tmp/diff_report.html', 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"{style_pass()} HTML file saved in ../tmp/diff_report.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--pkl_file", type=str, required=True, default="../tmp/diff.pkl",
                        help="pkl data file path of onnxruntime and r8 testing.")

    args = parser.parse_args()

    pkl_file = args.pkl_file

    main(pkl_file)

