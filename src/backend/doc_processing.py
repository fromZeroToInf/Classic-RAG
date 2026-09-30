import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

from backend.config import Chunking_Constants
from backend.config import settings
from backend.config import settings
from backend.models import Chunk, Doc_Hashes, MANIFEST_ADAPTER, Language
from common.getprojectroot import define_project_root_path
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument
from pathlib import Path
from pydantic import ValidationError
from tqdm.notebook import tqdm, trange
from transformers import AutoTokenizer
from typing import List, Tuple
import hashlib
import inspect
import json 
import jsonlines
import pprint
import re
import warnings
from typing import Callable



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

    def _get_normalizer(self, language: Language) -> Callable[[str], str]:
        match language:
            case Language.DE:
                return self._regex_normalize_text_ger
        raise ValueError(f"No normalizer for language {language!r}.")

    def _remove_processed_data(self, stem:str|None = None) -> None:
        """ If stem is None, chunks and output_docs will be completely deleted.
        """
        all_data = []
        if stem is None:
            chunks = list(settings.CHUNKS_OUT_DIR.glob("*.jsonl"))
            output_docs = list(settings.DOCUMENTS_OUT_DIR.glob("*.json"))
        else:
            chunks = list(settings.CHUNKS_OUT_DIR.glob(f"{stem}.jsonl"))
            output_docs = list(settings.DOCUMENTS_OUT_DIR.glob(f"{stem}.json"))
        
        all_data.extend( chunks + output_docs)
        if all_data:
            for fp in all_data:
                warnings.warn(f"removing processed data:\n{fp}")
                os.remove(fp)
                
    #TODO: if the manifest is empty -> remove all processed data
    def _manifest_load(self) -> dict[str,Doc_Hashes]:
        """Loads the manifest file. If no file or a corrupted file exists, all processed data will be removed.
        """
        
        path = list(settings.DOC_MANIFEST_OUT_DIR.glob("*.json"))
        if len(path) ==0:
            self._remove_processed_data()
            return {}
        else:
            path = path[0]
        
        try:
            return MANIFEST_ADAPTER.validate_json(path.read_bytes())
        except (ValueError, ValidationError):
            warnings.warn("Manifest file is corrupted. Removing files")
            self._remove_processed_data()
            path.touch()
            return {}
        
                
    def _manifest_cleaner(self, manifest: dict[str,Doc_Hashes], stem: str)-> None:
        """Removes stem from manifest and associated chunks and outputdocs"""
        manifest.pop(stem, None)
        warnings.warn(f"Updating manifest, file {stem} will be deleted.")
        self._remove_processed_data(stem=stem)
        self._manifest_save(manifest)
        
    def _manifest_save(self, manifest: dict[str, Doc_Hashes])->None:
        path = list(settings.DOC_MANIFEST_OUT_DIR.glob("*.json"))
        if not path:
            path = (settings.DOC_MANIFEST_OUT_DIR / "manifest.json")
            path.touch()
        else:
            path = path[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_bytes(MANIFEST_ADAPTER.dump_json(manifest, indent=2))
        tmp.replace(path)
    
    def _build_doc_hashes(self, stem:str, doc_id: str, language:Language=Language.DE) -> Doc_Hashes:
        """Creates the Doc_hash with meta_data"""
        normalizer = self._get_normalizer(language)
        dh = Doc_Hashes(
            source_name=stem,
            doc_id=doc_id,
            language = language,
            tokenizer=self._constants.TEXT_EMBEDDING_MODEL,
            max_tokens=self._constants.MAX_TOKENS,
            ceiling=self._constants.CEILING,
            threshold=self._constants.THRESHOLD_SMALL_CHUNKS,
            normalize_text=self._hash_fn(normalizer)
        )
        dh.total_hash = self._hash_dict(dh.model_dump(mode="json", exclude={"total_hash"}))
        return dh
    
    
    def _check_doc_hash_against_chunking_Constants(self, item:Doc_Hashes)-> bool:
        """Checks if the processed data was created by the actual config (chunking constants). In case of deviations the method will return False.
        """
        dh = Doc_Hashes(
                source_name=item.source_name,
                doc_id= item.doc_id,
                language = item.language,
                tokenizer=self._constants.TEXT_EMBEDDING_MODEL,
                max_tokens=self._constants.MAX_TOKENS,
                ceiling=self._constants.CEILING,
                threshold= self._constants.THRESHOLD_SMALL_CHUNKS,
                normalize_text= self._hash_fn(self._get_normalizer(item.language)),
                total_hash="",
            )
        dh.total_hash = self._hash_dict(dh.model_dump(mode="json", exclude={"total_hash"}))
        
        if dh.total_hash == item.total_hash:
            return True
        return False
        #5: tokenizer, max_tokens, ceiling, threshold, normalize_text
        # if item.ceiling != self._constants.CEILING:
        #     return False
        # if item.max_tokens != self._constants.MAX_TOKENS:
        #     return False
        # if item.normalize_text != self._hash_fn(self._get_normalizer(item.language)):
        #     return False
        # if item.threshold != self._constants.THRESHOLD_SMALL_CHUNKS:
        #     return False
        # if item.tokenizer != self._constants.TEXT_EMBEDDING_MODEL:
        #     return False
        # return True
    
    def _check_processed_docs_if_up_to_date(self)->dict[str,Doc_Hashes]:
        """ Checks the processed docs against the config. If a different config exists, the old processed docs need to be removed, and the processing needs to be restarted.
        """ 
        
        manifest = self._manifest_load()
        new_manifest = {}
        process_data = False
        if not manifest:
            self._remove_processed_data()
            
        for stem,doc_meta in manifest.items():
            if self._check_doc_hash_against_chunking_Constants(doc_meta) == False:
                self._remove_processed_data(stem=stem)
                process_data = True
            else:
                new_manifest[stem] = doc_meta
        if manifest != new_manifest:
            self._manifest_save(new_manifest)
        if process_data:
            self.read_documents_and_save()
            
        return new_manifest
    
    def _merge_chunks(self, a:Chunk, b:Chunk) -> Chunk:
        if a.doc_id != b.doc_id:
            raise ValueError(f"Cannot merge two chunks of different documents:\ndoc_id a:{a.doc_id}\ndoc_id b:{b.doc_id}")
        a = a.model_copy(deep=True)
        a.text = a.text + b.text
        a.contextualized_text = a.contextualized_text + b.contextualized_text
        a.meta.doc_items = a.meta.doc_items + b.meta.doc_items
        a.token_count = self._tokenizer.count_tokens(a.contextualized_text)
        return a

    def read_documents_and_save()-> None:
        fps = list(settings.DOCUMENTS_IN_DIR.glob("*.pdf"))

        converter = DocumentConverter()

        results = [(converter.convert(file), file.stem) for file in fps]
        total_status = [f"{f}: {res.status}" for res, f in results]

        settings.DOCUMENTS_OUT_DIR.mkdir(parents=True, exist_ok=True)

        for result, fn in results:
            doc = result.document
            data = doc.export_to_dict()
            fp = settings.DOCUMENTS_OUT_DIR / f"{fn}.json"
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        pprint.pprint(total_status)
    
    def make_chunks(self) -> None:
        """ Chunks will be written in settings.CHUNKS_OUT_DIR. See /src/backend/config.py.
        """
        hashes = self._hash_files(settings.DOCUMENTS_IN_DIR)

        
        #settings.CHUNKS_OUT_DIR.mkdir(parents=True, exist_ok=True)
        
        #load manifest to check doc is new
        manifest = self._check_processed_docs_if_up_to_date()
       
        for stem, doc_id in tqdm(hashes, desc="Reading Files", unit="file"):
            fp = list(settings.DOCUMENTS_OUT_DIR.glob(f"{stem}.json"))[0]
            
            doc_hash = self._build_doc_hashes(stem, doc_id)
            
            total_hash = self._hash_dict(doc_hash.model_dump())
            doc_hash.total_hash = total_hash
            
            # check if doc was already processed
            if stem in manifest:
                doc_meta = manifest[stem]
                if doc_hash.total_hash == doc_meta.total_hash:
                    continue
                
            manifest[stem] = doc_hash
            
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
        
        self._manifest_save(manifest)
    def load_chunks(self, one_file:Path|None=None) -> list[list[Chunk]]:
        
        chunks_path = list(settings.CHUNKS_OUT_DIR.glob("*.jsonl" if one_file==None else one_file))
        
        doc_chunks = []
        for fp in tqdm(chunks_path, desc="Reading files", unit="file"):
            with jsonlines.open(fp) as reader:
                loaded_chunks = [Chunk.model_validate(row) for row in tqdm(reader, desc="Reading chunks", unit="chunk", leave=False)]
                doc_chunks.append(loaded_chunks)
        
        return doc_chunks
        
            
    