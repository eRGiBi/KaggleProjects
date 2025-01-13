import json


class Params:
    def __init__(self, file_path=None):
        """
        Initialize Params object. Optionally load parameters from a JSON file.
        """
        self.params = {}
        if file_path:
            self.load(file_path)

    def load_from_json(self, file_path):
        """
        Load parameters from a JSON file.
        """
        with open(file_path, 'r') as f:
            self.params = json.load(f)

    def save_to_json(self, file_path):
        """
        Save parameters to a JSON file.
        """
        with open(file_path, 'w') as f:
            json.dump(self.params, f, indent=4)

    def from_dict(self, params_dict):
        """
        Update parameters from a dictionary.
        """
        self.params.update(params_dict)

    def __getitem__(self, key):
        """
        Get a parameter value.
        """
        return self.params.get(key)

    def __setitem__(self, key, value):
        """
        Set a parameter value.
        """
        self.params[key] = value

    def __repr__(self):
        """
        String representation of the Params object.
        """
        return f"Params({self.params})"
