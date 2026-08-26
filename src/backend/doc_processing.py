import hashlib
from pathlib import Path
from typing import List, Tuple

def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            hasher.update(block)
    return hasher.hexdigest()

def hash_files(dir_path:Path, type:str="pdf") -> List[Tuple[str,str]]:
    fps =  sorted(dir_path.glob(f"*.{type}"))
    return [(f.stem, hash_file(f)) for f in fps]
        