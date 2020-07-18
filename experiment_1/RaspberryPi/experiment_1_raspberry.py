import os
import numpy as np
import re
import time
from collections.abc import Iterable
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.comments import Comment
# import tflite_runtime.compat.v1 as tf
import tflite_runtime.interpreter as tflite

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_networks_data():
    out = []
    net_data = {'name': None, 'difficulty': None, 'precision': None}
    # in local file hierarchy it is in superior folder but in remote (due to copy script) file is in same directory
    file1 = open('/root/face-detection/experiment_1/networks_data.txt', 'r')
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
        # Load the TFLite model and allocate tensors.
        self.interpreter = tflite.Interpreter(model_path=f'{self.model_filepath}/{self.model_filepath.rsplit("/", 1)[1]}.tflite')
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Test the model on random input data.
        input_shape = input_details[0]['shape']
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        self.interpreter.set_tensor(input_details[0]['index'], input_data)

    def inference(self, infer_requests):
        times = [None] * infer_requests
        # do inference
        self.interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        print(output_data)
        for i in range(infer_requests):
            times[i], _ = record_time(self.interpreter.invoke, None)
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
            ws_data[f'A{row_nmbr}'] = 'Raspberry Pi: CPU'
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


def main():
    nns_dir = '/root/face-detection/models/mobilenet_v2'

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
        result['exec_t']['overall'], result['exec_t']['individual'] = record_time(cn.inference, 300)

        test_results.append(result)
    # if you change filename change it in 'copy_experiment_results.sh' script
    write_to_csv(test_results, 'res_exp_1.xlsx')


if __name__ == '__main__':
    # cProfile.run(main(sys.argv[1:]))
    main()
