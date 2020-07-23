from custom_exceptions import UnknownNetworkType


class NetworkType:
    #           data in order: (input_blob_index, output_blob_index)
    _known_networks = {'face-detection-0100': (0, 0),
                       'face-detection-0105': (0, 1)}

    def __init__(self, name: str):
        network_name = name.rsplit("/", 1)[1].rsplit('.', 1)[0]
        try:
            index = list(self._known_networks.keys()).index(network_name)
        except ValueError:
            raise UnknownNetworkType(f"Network: {network_name} is unknown. Please add support or change network.")
        self.network_type = list(NetworkType._known_networks.keys())[index]

    def get_input_blob_index(self):
        return NetworkType._known_networks[self.network_type][0]

    def get_output_blob_index(self):
        return NetworkType._known_networks[self.network_type][1]

    def get_name(self):
        return self.network_type

