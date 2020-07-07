import cv2
import numpy as np
from typing import Dict
from openvino.inference_engine import IECore


class NetworkModel:
    def __init__(self, model: IECore.IENetwork):
        self.model = model
        self.exec_model = None

    # if async supported add param for queue max size
    def deploy_network(self, device: str, ie_core: IECore):
        # num_requests=0 -> optimal nmbr of requests
        self.exec_model = ie_core.load_network(self.model, device, num_requests=0)
        self.model = None

    def sync_infer(self, input: Dict) -> Dict[str, np.ndarray]:
        # do the inference on first request (if more then one request do it in loop one infer per request)
        self.exec_model.requests[0].infer(input)
        return self.exec_model.requests[0].outputs
