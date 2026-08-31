import hashlib
import re
from pathlib import Path
from typing import List, Tuple

def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            hasher.update(block)
    return hasher.hexdigest()

def hash_files(dir_path:Path, type:str="pdf") -> List[Tuple[str,str]]:
    """
    Args:
        dir_path (Path): Directory Path
        type (str, optional): MIME Type. Defaults to "pdf".

    Returns:
        List[Tuple[str,str]]: [filename, hash]
    """
    fps =  sorted(dir_path.glob(f"*.{type}"))
    return [(f.stem, hash_file(f)) for f in fps]

def regex_normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text) # multiple whitespaces
    text = re.sub(r"\xad\s*", "", text) # one word separated by dash and whitespace
    return text
        