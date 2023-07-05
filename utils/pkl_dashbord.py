import os.path


def gen_mae_js_option(id: int=0):
    mae_option_template = \
        """ 
        <div id="c_$id$" style="width: 80%;height:500px;border-radius: 20px;background-color: white; margin: 10px auto"></div>
        <script type="text/javascript">
            var myChart = echarts.init(document.getElementById('c_$id$'));
            option_$id$ = {
                grid: { left: '15%', right: '15%', top: '10%', bottom: '10%'},
                title: {text: "onnxruntime vs rpprt_r8 diff", subtext: 'from input to layer output abs error', left: 'center'},
                xAxis: { type: 'category', data: $layer_names$, show: false, name: 'Layers'},
                yAxis: { type: 'value', name: 'Mean ABS Error' },
                toolbox: { show: true, feature: { magicType: { show: true, type: ['line', 'bar'] }, restore: { show: true }, saveAsImage: { show: true }}},
                tooltip: { trigger: 'axis', position: function (pt) { return [pt[0], '10%']; }},
                dataZoom: [{ type: 'inside', start: 0, end: 100 }, { start: 0, end: 100 }, {show: true, yAxisIndex: 0, filterMode: 'empty', width: 30, height: '80%', showDataShadow: false, left: '86%' }],
                series: [{data: $diff_values$, type: 'bar' }]
            };
            myChart.setOption(option_$id$);
        </script>
        """
    return mae_option_template.replace('$id$', str(id))

def gen_tensor_js_option(id: int=1):
    tensor_option_template = \
        """
        <div id="c_$id$" style="width: 80%;height:500px;border-radius: 20px;background-color: white; margin: 10px auto"></div>
        <script type="text/javascript">
            var myChart = echarts.init(document.getElementById('c_$id$'));
            option_$id$ = {
                title: {text: "$title$", left: 'center'},
                tooltip: {},
                xAxis: { type: 'category', data: $xaxis$ },
                yAxis: { type: 'category', data: $yaxis$ },
                dataZoom: [{ type: 'inside', start: 0, end: 100 }, { start: 0, end: 100 }, {show: true, yAxisIndex: 0, filterMode: 'empty', width: 30, height: '80%', showDataShadow: false, left: '96%' }],
                visualMap: { min: 0, max: $maxv$, calculable: true, realtime: false, inRange: { color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']}},
                series: [{ type: 'heatmap', data: $data_diff$, emphasis: { itemStyle: {borderColor: '#333', borderWidth: 1 }}, progressive: 1000, animation: false}]};
            myChart.setOption(option_$id$);
        </script>
        """
    return tensor_option_template.replace('$id$', str(id))

# xAxis, yAxis = [1,2,3,4 ...... ]
# data = [[0,0,1], [0,1,2] ...... []]


import numpy as np
from Pickle import load_pkl, check_pkl


def main(pkl_file: str = "diff.pkl"):

    if not os.path.exists(pkl_file):
        print("pkl file not found.")

    data = load_pkl(pkl_file)
    check_pkl(pkl_file)

    layer_names, diff_array, diff_values, shapes = [], [], [], []
    for layer_ix, v_dict in data.items():
        layer_names.append(f"{layer_ix}# {v_dict['name']}")
        diff_array.append(v_dict['diff'])
        diff_values.append(np.mean(v_dict['diff']))
        shapes.append(v_dict['shape'])
    print(shapes)
    #print(str(layer_names))
    #print(str(diff_values))
    option = gen_mae_js_option(id=0)
    option = option.replace("$layer_names$", str(layer_names))
    option_mae = option.replace("$diff_values$", str(diff_values))
    print(option)

    layers_num = len(data.items())
    tensor_options = []

    tensor_name = "/layer4/layer4.1/Add_output_0"
    for layer_ix, v_dict in data.items():
        if v_dict['name'] != tensor_name:
            continue
        shape = shapes[layer_ix]
        assert shape[0] == 1, "batch size must be 1"
        if len(shape) == 4:     # B,C,H,W
            channels = shape[1]
            wh = shape[2] * shape[3]
            diff = diff_array[layer_ix].reshape(channels, wh)
        elif len(shape) == 2:   # B, Class
            channels = 1
            wh = shape[-1]
            diff = diff_array[layer_ix].reshape(channels, wh)
        xaxis = [*range(wh)]
        yaxis = [*range(channels)]

        data_diff = []
        for xix in xaxis:
            for yix in yaxis:
                data_diff.append([xix, yix, diff[yix][xix]])
        maxv = np.max(diff)
        option = gen_tensor_js_option(id=1+layer_ix)
        option = option.replace('$xaxis$', str(xaxis)).replace('$yaxis$', str(yaxis)).replace('$maxv$', str(maxv)).replace('$data_diff$', str(data_diff)).replace('$title$', f"{tensor_name}, shape={shape}")
        tensor_options.append(option)



    assert len(tensor_options) == 1
    options_all = "\n".join([option_mae, option])



    with open('dashboard.html', 'r', encoding='utf-8') as f:
        html_template = f.read()
    html = html_template.replace("$options$", options_all)
    #print(html)

    with open('../tmp/diff_report.html', 'w', encoding='utf-8') as f:
        f.write(html)


if __name__ == "__main__":
    #pkl_file = input("pkl file path:\n")
    pkl_file = "diff.pkl"
    main(pkl_file)
