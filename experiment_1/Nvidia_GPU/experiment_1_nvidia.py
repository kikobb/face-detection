import os
import numpy as np
import re
import time
from collections.abc import Iterable
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.comments import Comment
import tensorflow.compat.v1 as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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

    for i, _ in enumerate(data):
        for measurement in data[i]['exec_t']['individual']:
            ws_data[f'A{row_nmbr}'] = 'GPU - Nvidia'
            ws_data[f'B{row_nmbr}'] = next(j for j, item in enumerate(g_mobilenet_data)
                                           if item["name"] == data[i]['network_name'])
            ws_data[f'C{row_nmbr}'] = int(round(data[i]['init_t'] * 1000000))
            ws_data[f'D{row_nmbr}'] = 0
            ws_data[f'E{row_nmbr}'] = int(round(data[i]['exec_t']['overall'] * 1000000))
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
    nns_dir = '/home/nvidia/models/mobilenet_v2'

    test_results = []
    count = 0
    for nn_dir in os.listdir(nns_dir):
        cn = CNN(f'{nns_dir}/{nn_dir}')
        print(f'{count} Network: {nn_dir}')
        count += 1
        result = {
            'network_name': nn_dir, 'init_t': None, 'load_t': None, 'exec_t':
                {
                    'overall': None, 'individual': None
                }
        }

        # load model form disk and initialize
        result['init_t'], _ = record_time(cn.init_graph, None)

        # perform inference
        result['exec_t']['overall'], result['exec_t']['individual'] = record_time(cn.inference, 30)

        test_results.append(result)
    # if you change filename change it in 'copy_experiment_results.sh' script
    write_to_csv(test_results, 'res_exp_1.xlsx')


if __name__ == '__main__':
    # cProfile.run(main(sys.argv[1:]))
    main()
