import pickle


def serialize_object(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
