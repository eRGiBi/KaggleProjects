
import pandas as pd


class ExperimentLogger:
    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, **kwargs):
        self.data.append(kwargs)

    def save(self, data):
        try:
            df = pd.read_csv(self.path)
            df.add(self.data)
            df.to_csv(self.path, index=False)

        except FileNotFoundError:
            print("File not found, creating new file")
            df = pd.DataFrame(self.data)
            df.to_csv(self.path, index=False)
