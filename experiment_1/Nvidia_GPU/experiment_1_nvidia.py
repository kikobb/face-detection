import os
import sys

import numpy as np
import re
import time
from collections.abc import Iterable
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.comments import Comment
import tensorflow.compat.v1 as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

g_default_number_of_inference_requests = 30

def get_infer_req_nmbr():
    if len(sys.argv) == 1 or len(sys.argv) > 2:
        return g_default_number_of_inference_requests
    if not sys.argv[1].isnumeric():
        return g_default_number_of_inference_requests
    return int(sys.argv[1])


def load_networks_data():
    out = []
    net_data = {'name': None, 'difficulty': None, 'precision': None}
    # in local file hierarchy it is in superior folder but in remote (due to copy script) file is in same directory
    file1 = open('/home/nvidia/networks_data.txt', 'r')
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


def record_time(func, params):
    ret = None
    start = time.perf_counter()
    if func is not None:
        if isinstance(params, Iterable):
            ret = func(*params)
        elif params is not None:
            ret = func(params)
        else:
            ret = func()
    stop = time.perf_counter()
    return stop - start, ret


class CNN(object):

    def __init__(self, model_filepath):
        # The file path of model
        self.model_filepath = model_filepath

    def init_graph(self):
        self.graph = tf.Graph()
        self.sess = tf.InteractiveSession(graph=self.graph)

        with tf.gfile.GFile(f'{self.model_filepath}/{self.model_filepath.rsplit("/", 1)[1]}_frozen.pb', 'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())

        # Define input tensor
        dimension = int(self.model_filepath.rsplit("/", 1)[1].rsplit('_', 1)[1])
        self.input = tf.placeholder(np.float32, shape=[None, dimension, dimension, 3], name='input')

        self.dummy_frame = np.full((1, dimension, dimension, 3), 0, dtype=int)
        tf.import_graph_def(self.graph_def, {'input': self.input})

    def inference(self, infer_requests):
        output_tensor = self.graph.get_tensor_by_name(f'{self.graph.get_operations()[-1].name}:0')
        times = [None] * infer_requests
        # do inference
        for i in range(infer_requests):
            times[i], _ = record_time(self.sess.run, (output_tensor, {self.input: self.dummy_frame}))
        self.sess.close()
        return times


def write_to_csv(data, file_name):
    # create excel workbook with one worksheet
    wb = Workbook()
    # access worksheet
    ws_data = wb.active
    # name ws
    ws_data.title = "main_data"
    # create header of spreadsheet
    ws_data['A1'] = 'Device'

    ws_data['B1'] = 'Net_name'
    ws_data['C1'] = 'Net_difficulty (in MACs (M))'
    ws_data['C1'].comment = Comment('MAC = Multiplication and Accumulation operation, (M) = in millions',
                                    'xbarna02')
    ws_data['D1'] = 'Net_precision (top 5)%'
    ws_data['D1'].comment = Comment('probability that correct answer occurs in top 5 predictions', 'xbarna02')
    ws_data['E1'] = 'Net_dropout'
    ws_data['F1'] = 'Net_input_dimension'
    ws_data['G1'] = 'Initialization (μs)'
    ws_data['H1'] = 'Loading (μs)'
    ws_data['I1'] = 'Overall_execution_of_one_batch (μs)'
    ws_data['J1'] = 'Individual_inference_execution (μs)'
    # make them bold
    for cell in ws_data["1:1"]:
        cell.font = Font(bold=True)
    # write data
    row_nmbr = 2

    for i, _ in enumerate(data):
        network = next(j for j, item in enumerate(g_mobilenet_data)
                       if item["name"] == data[i]['network_name'])
        for measurement in data[i]['exec_t']['individual']:
            ws_data[f'A{row_nmbr}'] = 'PC: GPU - Nvidia'
            ws_data[f'B{row_nmbr}'] = g_mobilenet_data[i]['name']
            ws_data[f'C{row_nmbr}'] = g_mobilenet_data[i]['difficulty']
            ws_data[f'D{row_nmbr}'] = g_mobilenet_data[i]['precision']
            ws_data[f'E{row_nmbr}'] = g_mobilenet_data[i]['name'].rsplit('_', 2)[1]
            ws_data[f'F{row_nmbr}'] = g_mobilenet_data[i]['name'].rsplit('_', 1)[1]
            ws_data[f'G{row_nmbr}'] = int(round(data[i]['init_t'] * 1000000))
            ws_data[f'H{row_nmbr}'] = 0
            ws_data[f'I{row_nmbr}'] = int(round(data[i]['exec_t']['overall'] * 1000000))
            ws_data[f'J{row_nmbr}'] = int(round(measurement * 1000000))

            row_nmbr += 1

    wb.save(filename=file_name)
    wb.close()

def main():
    nns_dir = '/home/nvidia/models/mobilenet_v2'

    test_results = []
    count = 0
    for nn_dir in g_mobilenet_data:
        cn = CNN(f'{nns_dir}/{nn_dir["name"]}')
        print(f'{count} Network: {nn_dir["name"]}')
        count += 1
        result = {
            'network_name': nn_dir["name"], 'init_t': None, 'load_t': None, 'exec_t':
                {
                    'overall': None, 'individual': None
                }
        }

        # load model form disk and initialize
        result['init_t'], _ = record_time(cn.init_graph, None)

        # perform inference
        result['exec_t']['overall'], result['exec_t']['individual'] = record_time(cn.inference, get_infer_req_nmbr())

        test_results.append(result)
    # if you change filename change it in 'download_experiment_results.sh' script
    write_to_csv(test_results, '/home/nvidia/res_exp_1.xlsx')


if __name__ == '__main__':
    # cProfile.run(main(sys.argv[1:]))
    main()
