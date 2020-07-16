import os
from openpyxl import Workbook
import numpy as np
import re
from openpyxl.styles import Font
from openpyxl.comments import Comment

import time
from openvino.inference_engine import IECore


def load_networks_data():
    out = []
    net_data = {'name': None, 'difficulty': None, 'precision': None}
    # in local file hierarchy it is in superior folder but in remote (due to copy script) file is in same directory
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
    # start = time.perf_counter()
    # do inference
    for i in range(infer_requests):
        times[i], _ = record_time(exec_net.infer, (inputs,))
    # stop = time.perf_counter()
    # info = time.get_clock_info('perf_counter')
    #
    # total_time = 0
    # for i, t in enumerate(times):
    #     total_time += t
    #     print(f'        infer {i}: {t}')
    # print(f'form partial: {total_time}')
    # print(f'    avg. per execution: {total_time / len(times)}')
    # print(f'execution: {stop - start}')
    # print(f'deference: {abs(total_time - (stop - start))}')
    # # print(f'sleep: {(stop - start)*1000000000} ns')
    # avg = sum(times) / len(times)
    return times


def record_time(func, params):
    ret = None
    start = time.perf_counter()
    if func is not None:
        # if
        ret = func(*params)
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
    ws_data['B1'] = 'Network_difficulty_level'
    ws_data['C1'] = 'Initialization (μs)'
    ws_data['D1'] = 'Loading (μs)'
    ws_data['E1'] = 'Overall_execution_of_one_batch (μs)'
    ws_data['F1'] = 'Individual_inference_execution (μs)'
    # make them bold
    for cell in ws_data["1:1"]:
        cell.font = Font(bold=True)
    # write data
    row_nmbr = 2
    for dev_name in data.keys():
        for i, _ in enumerate(data[dev_name]):
            for measurement in data[dev_name][i]['exec_t']['individual']:
                ws_data[f'A{row_nmbr}'] = dev_name
                ws_data[f'B{row_nmbr}'] = next(j for j, item in enumerate(g_mobilenet_data)
                                               if item["name"] == data[dev_name][i]['network_name'])
                ws_data[f'C{row_nmbr}'] = int(round(data[dev_name][i]['init_t'] * 1000000))
                ws_data[f'D{row_nmbr}'] = int(round(data[dev_name][i]['load_t'] * 1000000))
                ws_data[f'E{row_nmbr}'] = int(round(data[dev_name][i]['exec_t']['overall'] * 1000000))
                ws_data[f'F{row_nmbr}'] = int(round(measurement * 1000000))

                row_nmbr += 1

    # write data about networks
    # header
    ws_net_desc = wb.create_sheet("Network_diff")
    ws_net_desc['A1'] = 'Number'
    ws_net_desc['A1'].comment = Comment('this number pairs with table on \"main_data\" working sheet column '
                                        '\"Network_difficulty_level\"', 'xbarna02')
    ws_net_desc['B1'] = 'Network_name'
    ws_net_desc['C1'] = 'Difficulty (in MACs (M))'
    ws_net_desc['C1'].comment = Comment('MAC = Multiplication and Accumulation operation, (M) = in millions',
                                        'xbarna02')
    ws_net_desc['D1'] = 'Precision (top 5)%'
    ws_net_desc['D1'].comment = Comment('probability that correct answer occurs in top 5 predictions', 'xbarna02')
    for cell in ws_net_desc["1:1"]:
        cell.font = Font(bold=True)
    # data
    row_nmbr = 2
    for i, row in enumerate(g_mobilenet_data):
        ws_net_desc[f'A{row_nmbr}'] = i
        ws_net_desc[f'B{row_nmbr}'] = row['name']
        ws_net_desc[f'C{row_nmbr}'] = row['difficulty']
        ws_net_desc[f'D{row_nmbr}'] = row['precision']

        row_nmbr += 1

    wb.save(filename=file_name)


def main():
    # load plugin
    ie = IECore()
    # test_results = {'CPU': [], 'GPU': [], 'MYRIAD': []}
    test_results = {'CPU': []}
    nns_dir = '/home/openvino/face/models/mobilenet_v2'

    for device_name in test_results.keys():
        count = 0
        for nn_dir in os.listdir(nns_dir):
            print(f'{count} Device: {device_name}, Network: {nn_dir}')
            count += 1
            result = {
                'network_name': nn_dir, 'init_t': None, 'load_t': None, 'exec_t':
                    {
                        'overall': None, 'individual': None
                    }
            }
            # load model form disk and initialize
            result['init_t'], (input_blob, network) = record_time(
                init_and_read_graph, (
                    ie,
                    f'{nns_dir}/{nn_dir}/{nn_dir}_frozen.xml',
                    device_name
                ))

            # Load model to plugin ExecutableNetwork
            result['load_t'], exec_net = record_time(ie.load_network, (network, device_name))

            # perform inference
            result['exec_t']['overall'], result['exec_t']['individual'] = record_time(inference, (
                exec_net, input_blob, 3))

            test_results[device_name].append(result)

    # if you change filename change it in 'copy_experiment_results.sh' script
    write_to_csv(test_results, 'res_exp_1.xlsx')


if __name__ == '__main__':
    # cProfile.run(main(sys.argv[1:]))
    main()
