import onnx
from onnx import shape_inference, TensorProto
from onnx.helper import make_model, make_graph, make_tensor_value_info
import onnx_graphsurgeon as gs
from typing import List
import sys
import pathlib
sys.path.append(str(pathlib.Path(".").absolute().parent.parent))

from tools.OnnxModel import load_onnx, save_onnx, OnnxGraph
from tools.OnnxModel.TestModel import test_model_ort
from utils.StyleText import *


def modify_graph_surgeon(model: onnx.ModelProto, input_names: List[str] = [], output_names: List[str] = [],
                         rm_init: bool = True) -> onnx.ModelProto:
    """
    Based on NVIDIA TensorRT onnx-graphsurgeon.
    ** remove_initializer ** 使用 onnxruntime 一般是要移除 initializer 生成模型的，但是在 R8 上面有时移除 initializer 后的模型会
    无法运行，因此保留此参数，当出现这种情况时设置 remove_initializer=False
    """

    if len(input_names) == 0:
        input_names = [node.name for node in model.graph.input]
    if len(output_names) == 0:
        output_names = [node.name for node in model.graph.output]
    model = shape_inference.infer_shapes(model)

    print(f"{style_info()} Model input: {input_names}")
    print(f"{style_info()} Model output: {output_names}")

    graph = gs.import_onnx(model)
    tensors = graph.tensors()
    graph.inputs = [tensors[name] for name in input_names]
    graph.outputs = [tensors[name] for name in output_names]
    graph.cleanup()
    model = gs.export_onnx(graph)
    if rm_init:
        model = OnnxGraph.remove_initializer_from_input(model)
    print(f"{style_pass()} Modified graph finished.")
    return model


def _copy_initializers(source_model: onnx.ModelProto, target_model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Copy source_model's initializer into target_model, contains weight and bias of the model.
    """

    nodes, ix2output, output2ix = _get_node_sequence(target_model)
    require_weights_names = []
    for ix, node in nodes.items():
        node_inputs = (node['in'])[1:]
        require_weights_names.extend(node_inputs)

    for initializer in source_model.graph.initializer:

        if initializer.name in require_weights_names:
            new_initializer = onnx.TensorProto()
            new_initializer.name = initializer.name
            new_initializer.data_type = initializer.data_type
            new_initializer.dims.extend(initializer.dims)
            new_initializer.raw_data = initializer.raw_data
            target_model.graph.initializer.append(new_initializer)
    return target_model


def _get_node_sequence(model):
    graph = model.graph
    ix2output = {}
    output2ix = {}
    nodes = {}
    for ix, node in enumerate(graph.node):
        ix2output[ix] = (node.output[0])
        output2ix[node.output[0]] = ix
        nodes[ix] = {'ix': ix, 'node': node, 'in': node.input, 'out': node.output[0]}
    return nodes, ix2output, output2ix


def _get_producer(model, output_names):
    nodes, ix2output, output2ix = _get_node_sequence(model)

    Finish = []
    Stack = [output2ix[name] for name in output_names]
    Unvisited = [node[0] for node in nodes.items()]

    while len(Stack) > 0:

        checking_node_ix = Stack.pop()
        checking_node_input = nodes[checking_node_ix]['in']
        Finish.append(checking_node_ix)

        pop_ixs = []
        for ix in Unvisited:
            if nodes[ix]['out'] in checking_node_input:
                pop_ixs.append(ix)
                Stack.append(ix)
        Unvisited = list(set(Unvisited) - set(pop_ixs))

    return Finish


def modify_graph_onnxapi(model: onnx.ModelProto, input_names: List[str] = [],
                         output_names: List[str] = []) -> onnx.ModelProto:
    """
    Based on ONNX Python api.
    """

    if len(input_names) == 0:
        input_names = [node.name for node in model.graph.input]
    if len(output_names) == 0:
        output_names = [node.name for node in model.graph.output]
    print(f"{style_info()} Model input: {input_names}")
    print(f"{style_info()} Model output: {output_names}")

    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)

    ir_version = model.ir_version
    opset_version = model.opset_import[0].version

    nodes, ix2output, output2ix = _get_node_sequence(model)

    in_graph_inputs = [*set(input_names) - set([input_.name for input_ in model.graph.input])]
    org_graph_inputs = [*set(input_names) & set([input_.name for input_ in model.graph.input])]

    in_graph_outputs = [*set(output_names) - set([output_.name for output_ in model.graph.output])]
    org_graph_outputs = [*set(output_names) & set([output_.name for output_ in model.graph.output])]

    # get all producers of output-nodes and input-nodes
    output_producer = set(_get_producer(model, output_names))
    input_producer = set(_get_producer(model, in_graph_inputs))

    # clean up graph
    # 1. opt_nodes = output_producer - input_producer, but some nodes in input_producer is necessary in the graph
    # 2. check the deleted nodes in input_producer, if its output in opt_nodes' inputs, fetch it back
    # 3. all the nodes have checked
    opt_nodes = output_producer - input_producer
    flag = False
    while not flag:
        opt_in = list(nodes[ix]['in'] for ix in opt_nodes)
        opt_in = [item for sublist in opt_in for item in sublist]
        flag = True
        for node_ix in input_producer:
            if nodes[node_ix]['out'] in opt_in and nodes[node_ix]['out'] not in input_names:
                opt_nodes.add(node_ix)
                input_producer.remove(node_ix)
                flag = False
                break

    # construct model
    opt_input = []
    for name in in_graph_inputs:
        ix = output2ix[name]
        shape = model.graph.value_info[ix].type.tensor_type.shape
        shape = [dim.dim_value for dim in shape.dim]
        opt_input.append(make_tensor_value_info(name, TensorProto.FLOAT, shape))

    for name in org_graph_inputs:
        for input_ in model.graph.input:
            if input_.name == name:
                opt_input.append(input_)

    opt_output = []
    for name in in_graph_outputs:
        ix = output2ix[name]
        shape = model.graph.value_info[ix].type.tensor_type.shape
        shape = [dim.dim_value for dim in shape.dim]
        opt_output.append(make_tensor_value_info(name, TensorProto.FLOAT, shape))

    for name in org_graph_outputs:
        for output_ in model.graph.output:
            if output_.name == name:
                opt_output.append(output_)

    graph_opt = make_graph([*map(lambda x: nodes[x]['node'], opt_nodes)], 'lr', opt_input, opt_output)
    model_opt = make_model(graph_opt)

    # copy weights and bias data to model_opt
    model_opt = _copy_initializers(model, model_opt)
    model_opt.ir_version = ir_version
    model_opt.opset_import[0].version = opset_version

    # The graph & model may imcompleted because the wrong starting/ending points were chosen.
    # In this case, model_opt can still be created,
    # use check_model() to have a check and prevent the return of wrong model.
    model_opt = shape_inference.infer_shapes(model_opt)

    try:
        onnx.checker.check_model(model_opt)
    except Exception as e:
        print(e)
        print(f"{style_warning()} Please check model in onnxruntime or use onnx-graphsurgeon.")
    print(f"{style_pass()} Modified graph finished.")

    return model_opt


if __name__ == "__main__":
    model = load_onnx("../../models/resnet50.onnx")

    input_names = ['data']
    output_names = ['output']

    # onnx-graphsurgeon version
    print("Modifying model with onnx-graphsurgeon version ... ")
    model_opt = modify_graph_surgeon(model, input_names=input_names, output_names=output_names)
    save_onnx(model_opt, "../../models/subgraph_surgeon.onnx")
    print("Testing model ...")
    test_model_ort("../../models/subgraph_surgeon.onnx")

    print("\n\n")

    # onnx api version
    print("Modifying model with onnx api version ... ")
    model_opt = modify_graph_onnxapi(model, input_names=input_names, output_names=output_names)
    save_onnx(model_opt, "../../models/subgraph_api.onnx")
    print("Testing model ...")
    test_model_ort("../../models/subgraph_api.onnx")



