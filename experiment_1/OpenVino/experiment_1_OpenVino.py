import os
import sys

import numpy as np
import re
from collections.abc import Iterable
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.comments import Comment

import time
from openvino.inference_engine import IECore


def is_raspberry():
    if len(sys.argv) == 2 and (sys.argv[1] != 'raspberry' or sys.argv[1] != 'pi'):
        return True
    return False


def load_networks_data():
    out = []
    net_data = {'name': None, 'difficulty': None, 'precision': None}
    # in local file hierarchy it is in superior folder but in remote (due to copy script) file is in same directory
    if is_raspberry():
        file1 = open('/root/face-detection/experiment_1/networks_data.txt', 'r')
    else:
        file1 = open('networks_data.txt', 'r')
    lines = file1.readlines()
    for i, line in enumerate(lines):
        if line == '\n':
            break
        net_data = dict()
        net_data['name'], net_data['difficulty'], net_data['precision'] = re.split('[;]', line[:-1])
        out.append(net_data)
    out = sorted(out, key=lambda k: int(k['difficulty']), reverse=True)
    return out


g_mobilenet_data = load_networks_data()


def init_and_read_graph(ie, model_filepath, device_name):
    # read model IRs
    net_mobilenet = ie.read_network(model='{0}'.format(model_filepath),
                                    weights='{0}.bin'.format(model_filepath[:-4]))

    # I/O blobs
    # (aux) net_face_detect.inputs -> dict of DataPtr obj
    # (aux) described in model - different types of input sources for NN
    # (aux) get the keyword of first element in dict
    input_blob = list(net_mobilenet.inputs)[0]
    out_blob = list(net_mobilenet.outputs)[0]

    # pre process image input
    n, c, h, w = net_mobilenet.inputs[input_blob].shape

    dummy_frame = np.full((c, h, w), 0, dtype=int)

    configure_plugins(ie, device_name)

    return {input_blob: dummy_frame}, net_mobilenet


def configure_plugins(ie, device_name):
    # set up plugins
    if device_name == 'CPU':
        # CPU
        ie.set_config(config={
            "CPU_THROUGHPUT_STREAMS": "1",
            "CPU_THREADS_NUM": "8",
            "CPU_BIND_THREAD": "YES",
        }, device_name='CPU')
    elif device_name == 'GPU':
        # GPU
        ie.set_config(config={"GPU_THROUGHPUT_STREAMS": "1"}, device_name='GPU')
    elif device_name == 'MYRIAD':
        pass


def inference(exec_net, inputs, infer_requests):
    times = [None] * infer_requests
    # do inference
    for i in range(infer_requests):
        times[i], _ = record_time(exec_net.infer, (inputs,))
    return times


def record_time(func, params):
    ret = None
    start = time.perf_counter()
    if isinstance(params, Iterable):
        ret = func(*params)
    elif params is not None:
        ret = func(params)
    else:
        ret = func()
    stop = time.perf_counter()
    return stop - start, ret


def write_to_csv(data, file_name):
    # create excel workbook with one worksheet
    wb = Workbook()
    # access worksheet
    ws_data = wb.active
    # name ws
    ws_data.title = "main_data"
    # create header of spreadsheet
    ws_data['A1'] = 'Device'
    ws_data['A1'].comment = Comment('HW zariadenie', 'xbarna02')
    ws_data['B1'] = 'Network_name'
    ws_data['B1'].comment = Comment('nazov neuronovej siete', 'xbarna02')
    ws_data['C1'] = 'Difficulty (in MACs (M))'
    ws_data['C1'].comment = Comment('obtiaznost v MAC = Multiplication and Accumulation operation, (M) = in millions',
                                    'xbarna02')
    ws_data['D1'] = 'Precision (top 5)%'
    ws_data['D1'].comment = Comment('presnost, probability that correct answer occurs in top 5 predictions', 'xbarna02')
    ws_data['E1'] = 'Net_dropout'
    ws_data['E1'].comment = Comment('koeficient zjednodusenia siete', 'xbarna02')
    ws_data['F1'] = 'Net_input_dimension'
    ws_data['F1'].comment = Comment('dimenzie vstupneho obrazku', 'xbarna02')
    ws_data['G1'] = 'Initialization (μs)'
    ws_data['G1'].comment = Comment('inicializacia siete', 'xbarna02')
    ws_data['H1'] = 'Loading (μs)'
    ws_data['H1'].comment = Comment('doba nacitania siete do zariadenie', 'xbarna02')
    ws_data['I1'] = 'Overall_execution_of_one_batch (μs)'
    ws_data['I1'].comment = Comment('celkova doba spustenia jedneho test', 'xbarna02')
    ws_data['J1'] = 'Individual_inference_execution (μs)'
    ws_data['J1'].comment = Comment('doba spustenia jednotlivej inferencie', 'xbarna02')
    # make them bold
    for cell in ws_data["1:1"]:
        cell.font = Font(bold=True)
    # write data
    row_nmbr = 2
    for dev_name in data.keys():
        for i, _ in enumerate(data[dev_name]):
            network = next(j for j, item in enumerate(g_mobilenet_data)
                           if item["name"] == data[dev_name][i]['network_name'])
            for measurement in data[dev_name][i]['exec_t']['individual']:
                ws_data[f'A{row_nmbr}'] = f'{"Raspberry Pi" if is_raspberry() else "PC"}: {dev_name if dev_name != "GPU" else f"{dev_name} - Intel"}'
                ws_data[f'B{row_nmbr}'] = g_mobilenet_data[i]['name']
                ws_data[f'C{row_nmbr}'] = g_mobilenet_data[i]['difficulty']
                ws_data[f'D{row_nmbr}'] = g_mobilenet_data[i]['precision']
                ws_data[f'E{row_nmbr}'] = g_mobilenet_data[i]['name'].rsplit('_', 2)[1]
                ws_data[f'F{row_nmbr}'] = g_mobilenet_data[i]['name'].rsplit('_', 1)[1]
                ws_data[f'G{row_nmbr}'] = int(round(data[dev_name][i]['init_t'] * 1000000))
                ws_data[f'H{row_nmbr}'] = int(round(data[dev_name][i]['load_t'] * 1000000))
                ws_data[f'I{row_nmbr}'] = int(round(data[dev_name][i]['exec_t']['overall'] * 1000000))
                ws_data[f'J{row_nmbr}'] = int(round(measurement * 1000000))

                row_nmbr += 1

    wb.save(filename=file_name)


def main():
    raspberry = is_raspberry()
    # load plugin
    ie = IECore()
    test_results = {'CPU': [], 'MYRIAD': []}
    if not raspberry:
        test_results['GPU'] = []
    # test_results = {'GPU': []}
    # test_results = {'CPU': []}
    nns_dir = '/home/openvino/face/models/mobilenet_v2'
    if raspberry:
        nns_dir = '/root/face-detection/model_library/mobilenet_v2'
    for device_name in test_results.keys():
        count = 0
        for nn_dir in g_mobilenet_data:
            print(f'{count} Device: {device_name}, Network: {nn_dir["name"]}')
            count += 1
            result = {
                'network_name': nn_dir["name"], 'init_t': None, 'load_t': None, 'exec_t':
                    {
                        'overall': None, 'individual': None
                    }
            }
            # load model form disk and initialize
            result['init_t'], (input_blob, network) = record_time(
                init_and_read_graph, (
                    ie,
                    f'{nns_dir}/{nn_dir["name"]}/{nn_dir["name"]}_frozen.xml',
                    device_name
                ))

            # Load model to plugin ExecutableNetwork
            result['load_t'], exec_net = record_time(ie.load_network, (network, device_name))

            # perform inference
            print('inference_begin')
            result['exec_t']['overall'], result['exec_t']['individual'] = record_time(inference, (
                exec_net, input_blob, 10))
            print('inference_end')
            test_results[device_name].append(result)

    # if you change filename change it in 'copy_experiment_results.sh' script
    write_to_csv(test_results, 'res_exp_1.xlsx')


if __name__ == '__main__':
    # cProfile.run(main(sys.argv[1:]))
    main()
