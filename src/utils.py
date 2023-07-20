import pickle


def serialize_object(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)
