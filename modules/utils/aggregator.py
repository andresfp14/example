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

    # 1) Track which folder these files came from
    data_dict["_directory"] = directory_path

    # 2) Load YAML files (resolve OmegaConf where possible)
    for file_path in base_path.rglob("*.yaml"):
        content = OmegaConf.load(str(file_path))
        key = file_path.relative_to(base_path).as_posix().replace("\\", "/").replace("/", ".").replace(file_path.suffix, "")
        try:
            data_dict[key] = OmegaConf.to_container(content, resolve=True)
        except Exception:
            data_dict[key] = OmegaConf.to_container(content, resolve=False)

    # 3) Load JSON files
    for file_path in base_path.rglob("*.json"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        key = file_path.relative_to(base_path).as_posix().replace("\\", "/").replace("/", ".").replace(file_path.suffix, "")
        data_dict[key] = content

    # 4) Record CSV file paths (left as strings for later pandas loading)
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
        # 1) Map each subfolder to load_files
        # 2) Wrap with tqdm for visibility when many runs exist
        results = list(
            tqdm(
                p.imap(load_files, subfolders),
                total=len(subfolders),
                desc="Loading folders",
            )
        )

    return results
