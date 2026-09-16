import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import hashlib
import re
from pathlib import Path
from typing import List, Tuple
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from backend.models import Chunk
from backend.models import Doc_Hashes
from backend.config import Chunking_Constants
from backend.config import settings
import jsonlines
from tqdm.notebook import tqdm, trange
import inspect
import json 

class Doc_Processing:
    def __init__(self, chunking_constants: Chunking_Constants, merge_peers=True, dev_mode=False):
        self._constants = chunking_constants
        
        if not dev_mode:
            self._tokenizer = HuggingFaceTokenizer(
                        tokenizer=AutoTokenizer.from_pretrained(self._constants.TEXT_EMBEDDING_MODEL),
                        max_tokens=self._constants.MAX_TOKENS
                    )
            self._hyb_chunker = HybridChunker(
                    tokenizer=self._tokenizer,
                    merge_peers=merge_peers,
                )
        
    def update_constants(self, chunking_constants: Chunking_Constants):
        self._constants = chunking_constants
    
    def _hash_file(self, path: Path) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                hasher.update(block)
        return hasher.hexdigest()

    def _hash_files(self, dir_path:Path, type:str="pdf") -> List[Tuple[str,str]]:
        """
        Args:
            dir_path (Path): Directory Path
            type (str, optional): MIME Type. Defaults to "pdf".

        Returns:
            List[Tuple[str,str]]: [filename, doc_id:hash]
        """
        fps =  sorted(dir_path.glob(f"*.{type}"))
        return [(f.stem, self._hash_file(f)) for f in fps]

    def _hash_fn(self, fn) -> str:
        src = inspect.getsource(fn)
        return hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]
    
    def _hash_dict(self, d:dict) -> str:
        json_obj = json.dumps(d, sort_keys=True)
        return hashlib.sha256(json_obj.encode("utf-8")).hexdigest()[:12]
    
    def _regex_normalize_text_ger(self,text: str) -> str:
        """Removes multiple whitespaces and word separation by one dash and whitespaces
        """
        text = re.sub(r"[ \t\r\f\v]+", " ", text) # multiple whitespaces
        text = re.sub(r"(?<=\w)\xad\n", "", text) # one word separated by dash and newline
        text = re.sub(r"(?<=\w)\xad (?!und)", "", text) #one word separated by dash and whitespace
        return text

    def _merge_chunks(self, a:Chunk, b:Chunk) -> Chunk:
        if a.doc_id != b.doc_id:
            raise ValueError(f"Cannot merge two chunks of different documents:\ndoc_id a:{a.doc_id}\ndoc_id b:{b.doc_id}")
        a = a.model_copy(deep=True)
        a.text = a.text + b.text
        a.contextualized_text = a.contextualized_text + b.contextualized_text
        a.meta.doc_items = a.meta.doc_items + b.meta.doc_items
        a.token_count = self._tokenizer.count_tokens(a.contextualized_text)
        return a


    def make_chunks(self) -> None:
        """ Chunks will be written in settings.CHUNKS_OUT_DIR. See /src/backend/config.py.
        """
        hashes = self._hash_files(settings.DOCUMENTS_IN_DIR)

        
        #settings.CHUNKS_OUT_DIR.mkdir(parents=True, exist_ok=True)
        
        #load manifest to check doc is new
        manifest:dict[str,Doc_Hashes] = {}
        with jsonlines.open(settings.DOC_HASHES_OUT_DIR, mode="r") as reader:
            for key,value in tqdm(reader, desc="Loading Manifest", unit="file"):
                manifest[key] = value
       
        for stem, doc_id in tqdm(hashes, desc="Reading Files", unit="file"):
            fp = list(settings.DOCUMENTS_OUT_DIR.glob(f"{stem}.json"))[0]
            
                
            
            doc_hash = Doc_Hashes(source_name=stem, doc_id=doc_id, tokenizer=self._constants.TEXT_EMBEDDING_MODEL, max_tokens=self._constants.MAX_TOKENS, ceiling=self._constants.CEILING, threshold=self._constants.THRESHOLD_SMALL_CHUNKS, normalize_text=self._hash_fn(self._regex_normalize_text_ger))
            
            total_hash = self._hash_dict(doc_hash.model_dump())
            doc_hash.total_hash = total_hash
            
            # TODO: check if doc was already processed
            if stem in manifest:
                doc_info_manif = manifest[stem]
                if doc_hash.total_hash == doc_info_manif.total_hash:
                    continue
                #case wipe all
            doc_mapping = {stem: doc_hash}
            
            with jsonlines.open(settings.DOC_HASHES_OUT_DIR, mode="w") as writer:
                writer.write(json.dumps(doc_mapping))
            
            
            doc = DoclingDocument.load_from_json(fp)
            out_fp = settings.CHUNKS_OUT_DIR / f"{stem}.jsonl"
            with jsonlines.open(out_fp, mode="w") as writer:
                chunk_no = 0
                chunk_buffer: Chunk | None = None
                
                for chunk in tqdm(self._hyb_chunker.chunk(dl_doc=doc), desc="Processing chunks", unit="chunk", leave=False ):
                    chunk.text = self._regex_normalize_text_ger(chunk.text)
                    ctx = self._hyb_chunker.contextualize(chunk)
                    c = Chunk(
                        chunk_id="",
                        doc_id=doc_id,
                        doc_name=str(stem)+".pdf",
                        text=chunk.text,
                        contextualized_text=ctx,
                        token_count=self._tokenizer.count_tokens(ctx),
                        meta=chunk.meta
                    )
                    if chunk_buffer is None:
                        chunk_buffer = c
                        continue
                    
                    is_small = ((chunk_buffer.token_count < self._constants.THRESHOLD_SMALL_CHUNKS) or 
                                (c.token_count < self._constants.THRESHOLD_SMALL_CHUNKS))
                    
                    fits = chunk_buffer.token_count + c.token_count < self._constants.CEILING
                    if is_small and fits:
                        chunk_buffer = self._merge_chunks(chunk_buffer, c)
                    else:
                        chunk_buffer.chunk_id= str(chunk_no)
                        writer.write(chunk_buffer.model_dump(mode="json"))
                        chunk_no += 1
                        chunk_buffer  = c
            
                if chunk_buffer is not None:
                    chunk_buffer.chunk_id = str(chunk_no)
                    writer.write(chunk_buffer.model_dump(mode="json"))
                    chunk_no +=1

    def load_chunks(self, one_file:Path|None=None) -> list[list[Chunk]]:
        
        chunks_path = list(settings.CHUNKS_OUT_DIR.glob("*.jsonl" if one_file==None else one_file))
        
        doc_chunks = []
        for fp in tqdm(chunks_path, desc="Reading files", unit="file"):
            with jsonlines.open(fp) as reader:
                loaded_chunks = [Chunk.model_validate(row) for row in tqdm(reader, desc="Reading chunks", unit="chunk", leave=False)]
                doc_chunks.append(loaded_chunks)
        
        return doc_chunks
        
            
    