import numpy as np

class IndexedArray:

    def __init__(self, input_dict):
        
        self.key_to_index = {}
        array_list = []

        for idx, (key, array) in enumerate(input_dict.items()):
            self.key_to_index[key] = idx
            array_list.append(np.array(array, dtype=float).reshape(1,-1))

        self.array = np.vstack(array_list)

    def __getitem__(self, key):
        
        #if string, return the corresponding row
        if isinstance(key, str):
            return self.array[self.key_to_index[key]]
        #if list of strings, return the corresponding rows
        elif isinstance(key, list):
            return self.array[[self.key_to_index[name] for name in key]]
        #if numpy array, send to list of strings and then return the corresponding rows
        elif isinstance(key, np.ndarray):
            return self.array[[self.key_to_index[str(name)] for name in key]]
        else:
            raise TypeError('Invalid argument type.')
        
    def get_all_keys(self):
        return list(self.key_to_index.keys())