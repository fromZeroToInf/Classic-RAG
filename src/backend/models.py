from pydantic import BaseModel,Field
from docling_core.transforms.chunker.doc_chunk import DocMeta

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    text: str
    contextualized_text: str
    token_count: int
    meta: DocMeta

class Doc_Hashes(BaseModel):
    source_name: str
    doc_id: str
    tokenizer: str
    max_tokens: str
    ceiling: int
    threshold: int
    normalize_text: str
    total_hash:str|None = Field(default=None)
    