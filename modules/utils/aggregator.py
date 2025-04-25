import json
from pathlib import Path
from omegaconf import OmegaConf
from multiprocessing import Pool
from tqdm import tqdm
import modules.utils.hydraqol as hq

def load_files(directory_path):
    """
    Globs all YAML, JSON, and CSV files in the given directory path.
    Loads YAML/JSON content into a dict, stores CSV path as is.
    Returns a dictionary of the form:
       {
         "filename": yaml_content,
         "filename": json_content,
         "filename": "path/to/filename.csv",
         ...
       }
    """
    base_path = Path(directory_path)
    data_dict = {}

    # directory_path = str(directory_path).replace("\\", "/")
    data_dict["_directory"] = directory_path

    # Load yaml files
    for file_path in base_path.rglob("*.yaml"):
        content = OmegaConf.load(str(file_path))
        key = file_path.relative_to(base_path).as_posix().replace("\\", "/").replace("/", ".").replace(file_path.suffix, "")
        try:
            data_dict[key] = OmegaConf.to_container(content, resolve=True)
        except Exception as e:
            data_dict[key] = OmegaConf.to_container(content, resolve=False)

    # Load json files
    for file_path in base_path.rglob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        key = file_path.relative_to(base_path).as_posix().replace("\\", "/").replace("/", ".").replace(file_path.suffix, "")
        data_dict[key] = content

    # Load csv files
    for file_path in base_path.rglob("*.csv"):
        key = file_path.relative_to(base_path).as_posix().replace("\\", "/").replace("/", ".").replace(file_path.suffix, "")
        data_dict[key] = str(file_path)
    
    return data_dict

def load_folders(base_dir, max_pool=8):
    """
    Iterates over each subfolder in `base_dir`, uses multiprocessing
    to load files (YAML, JSON, CSV) from each subfolder in parallel.
    Returns a list of dicts, each dict containing:
       - Loaded file data keyed by filename
       - A special field `_directory` with the subfolder name/relative path
    """
    base_path = Path(base_dir)
    subfolders = [p for p in base_path.iterdir() if p.is_dir()]

    # Use multiprocessing Pool
    with Pool(processes=max_pool) as p:
        # Map the list of subfolders to the _load_subfolder function
        # and wrap with tqdm for a progress bar
        results = list(
            tqdm(
                p.imap(load_files, subfolders),
                total=len(subfolders),
                desc="Loading folders",
            )
        )

    return results
