import json
import yaml
from pathlib import Path

def load_config(path):
    """
    Reads a JSON or YAML file and returns its contents as a dictionary.

    Args:
        path (str): Path to the JSON or YAML file. The function determines the file type based on its extension (.json, .yaml, .yml).

    Returns:
        dict: A dictionary containing the file's contents. In case of an error (e.g., file not found, incorrect format), returns a dictionary with an 'error' key and an error message as its value.

    Example:
        file_contents = read_json_or_yaml('/path/to/file.json')
        # or
        file_contents = read_json_or_yaml('/path/to/file.yaml')
    """
    # Check if the file is a JSON file based on its extension
    if path.endswith('.json'):
        with open(path, 'r') as file:
            # Load and return the JSON file content
            return json.load(file)
    # Check if the file is a YAML file based on its extension
    elif path.endswith(('.yaml', '.yml')):
        with open(path, 'r') as file:
            # Load and return the YAML file content
            return yaml.safe_load(file)
    else:
        # Raise an error if the file format is neither JSON nor YAML
        raise ValueError("Unsupported file format. Please provide a JSON or YAML file.")

def save_config(config, path, format='json'):
    """
    Saves a configuration dictionary as a JSON or YAML file.

    Args:
        config (dict): The configuration dictionary to be saved.
        path (str): Path where the file will be saved. The file extension should match the desired format.
        format (str, optional): Format of the file to save ('json' or 'yaml'). Default is 'json'.

    Returns:
        bool: True if the file was saved successfully, False otherwise.

    Raises:
        ValueError: If the format is not 'json' or 'yaml'.

    Example:
        config = {
            'setting1': 'value1',
            'setting2': 'value2'
        }
        save_config_as_json_or_yaml(config, '/path/to/config.json')
        # or
        save_config_as_json_or_yaml(config, '/path/to/config.yaml', format='yaml')
    """
    if format == 'json':
        with open(path, 'w') as file:
            json.dump(config, file, indent=4)
    elif format == 'yaml':
        with open(path, 'w') as file:
            yaml.dump(config, file)
    else:
        raise ValueError("Unsupported format. Please choose 'json' or 'yaml'.")

def multi_load(base_path="./data/configs", format='json', **kwargs):
    """
    Loads multiple configuration files based on a base path and a set of key-value pairs.

    Parameters:
    base_path (str): The base directory path where configuration files are stored.
    **kwargs: Key-value pairs where each key is the prefix of the configuration type 
              and the value is the specific configuration name. The function constructs 
              file paths based on these pairs.

    Returns:
    dict: A dictionary where each key is the same as the keys provided in kwargs, and 
          each value is the loaded configuration from the corresponding JSON file. 
          If a file is not found, the value will be None.

    Example:
    configs = multi_load(base_path="./data/configs", model="resnet18", dataset="AB")
    """
    configs = {}

    for k, v in kwargs.items():
        # Construct the file path using pathlib with f-string
        file_path = Path(base_path) / f"{k}s" / f"{k}_{v}.{format}"

        # Load the JSON file
        try:
            with open(file_path, 'r') as file:
                if format == 'json':
                    configs[k] = json.load(file)
                elif format in ['yaml', 'yml']:
                    configs[k] = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            configs[k] = None
        except ValueError:
            print(f"Unsupported format: {format}")
            configs[k] = None

    return configs

def multi_save(save_path="./data/configs", format='json', **kwargs):
    """
    Saves multiple configuration files to a specified path.

    Parameters:
    save_path (str): The directory path where the configuration files will be saved.
    **kwargs: Key-value pairs where each key is the prefix of the configuration type 
              and the value is the configuration data to be saved. The function constructs 
              file names based on these pairs and saves them as JSON files.

    Example:
    multi_save(save_path="./data/configs", model=model_config, dataset=dataset_config)
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

    for k, v in kwargs.items():
        file_path = save_path / f"{k}_{v}.{format}"

        try:
            with open(file_path, 'w') as file:
                if format == 'json':
                    json.dump(v, file, indent=4)
                elif format in ['yaml', 'yml']:
                    yaml.dump(v, file)
        except ValueError:
            print(f"Unsupported format: {format}")