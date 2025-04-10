import pickle
import pandas as pd

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

class PDStatistics_calculator(PickleSerializable):
    def __init__(self):
        self.sample_attributes = pd.DataFrame()
        self.annotation_attributes = pd.DataFrame()
        
    def add_sample_attributes(self, item_sample:dict):
        sample = pd.DataFrame([item_sample])
        self.sample_attributes = pd.concat([self.sample_attributes, sample], ignore_index=True)

    def add_annotation_attributes(self, annotation_sample:list):
        annotation = pd.DataFrame(annotation_sample)
        self.annotation_attributes = pd.concat([self.annotation_attributes, annotation], ignore_index=True)
