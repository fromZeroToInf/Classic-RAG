from pydantic import BaseModel,Field, TypeAdapter
from docling_core.transforms.chunker.doc_chunk import DocMeta
from enum import StrEnum
from typing import Callable

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    text: str
    contextualized_text: str
    token_count: int
    meta: DocMeta

class Language(StrEnum):
    DE = "de"
    EN = "en"
    
class Doc_Hashes(BaseModel):
    """
    Meta infos used for ingestion caching.
    Doc_hashes is a part of the manifest.
    """
    source_name: str
    doc_id: str
    language: Language
    tokenizer: str
    max_tokens: int
    ceiling: int
    threshold: int
    normalize_text: str
    total_hash:str|None = Field(default=None)
    

"""
MANIFEST: dict[stem, doc_meta]
"""
MANIFEST_ADAPTER = TypeAdapter(dict[str, Doc_Hashes])
