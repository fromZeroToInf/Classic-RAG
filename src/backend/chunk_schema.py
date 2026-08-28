from pydantic import BaseModel
from docling_core.transforms.chunker.doc_chunk import DocMeta

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    contextualized_text: str
    token_count: int
    meta: DocMeta

