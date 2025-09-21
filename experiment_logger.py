import os
import csv
import json


class ExperimentLogger:

    def __init__(self, param_path=None, result_path=None):
        """Initialize Params object. Optionally load parameters from a JSON file."""
        self.params = {}
        self.data = []
        self.param_path = param_path
        self.result_path = result_path

    def load_from_json(self, file_path):
        """Load parameters from a JSON file."""
        with open(file_path, 'r') as f:
            self.params = json.load(f)

    def save_to_json(self):
        """Save parameters to a JSON file."""
        with open(self.param_path, 'w', encoding="utf8") as f:
            json.dump(self.params, f, indent=4)

    def save_to_csv(self, data):
        file_exists = os.path.exists(self.param_path)

        with open(self.param_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(data)

    def save_results(self, result_dict):
        with open(self.result_path, 'wb') as f:
            json.dump(self.params, f, indent=4)

    def add_from_dict(self, params_dict):
        """Update parameters from a dictionary."""
        self.params.update(params_dict)

    def log(self, **kwargs):
        self.data.append(kwargs)

    def __getitem__(self, key):
        """Get a parameter value."""
        return self.params.get(key)

    def __setitem__(self, key, value):
        """Set a parameter value."""
        self.params[key] = value

    def __repr__(self):
        """String representation of the Params object."""
        return f"Params({self.params})"
