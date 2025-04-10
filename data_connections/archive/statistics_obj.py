import pickle

class PickleSerializable:
    def save(self, filename):
        """Save the object as a pickle file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """Load an object from a pickle file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)

class Statistics_obj(PickleSerializable):
    def __init__(self, attributes:list, data:list):
        self.attributes = attributes
        self.data = data
        