import pickle

def save_object(obj, path):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_object(path):
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)
    return obj