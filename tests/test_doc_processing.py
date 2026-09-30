import pytest
from backend.doc_processing import Doc_Processing
from backend.config import Chunking_Constants
from backend.models import Chunk, Doc_Hashes,MANIFEST_ADAPTER
from backend.config import settings
from pathlib import Path
import json

@pytest.fixture
def cts():
 return Chunking_Constants()

@pytest.fixture
def dp(cts):
    return Doc_Processing(cts, dev_mode=True)

@pytest.fixture
def out_dirs(tmp_path, monkeypatch)-> tuple[Path, Path]:
    chunks = tmp_path / "chunks"
    docs = tmp_path / "documents"
    chunks.mkdir()
    docs.mkdir()
    monkeypatch.setattr(settings, "CHUNKS_OUT_DIR", chunks)
    monkeypatch.setattr(settings, "DOCUMENTS_OUT_DIR", docs)
    return chunks, docs

@pytest.fixture
def mock_settings(tmp_path, monkeypatch)-> Path:
    manifest = tmp_path / "manifest"
    manifest.mkdir()
    monkeypatch.setattr(settings, "DOC_MANIFEST_OUT_DIR", manifest)
    return manifest

def test_regex_normalize_text_mult_whitespaces(dp) -> None:
    text1= "H  E  L  L  O"
    text2= "H E L L O"
    text3= "  HELLO  "
    res = "HELLO"
    
    res1 = dp._regex_normalize_text_ger(text1)
    res2 = dp._regex_normalize_text_ger(text2)
    res3 = dp._regex_normalize_text_ger(text3)
    
    assert res1 == text2
    assert res2 == text2
    assert res3 == " "+res + " "

def test_regex_normalize_text_mult_whitespaces(dp) -> None:
    text2= "HE\xad LLO"
    text3= "HE\xad\nLLO"
    text4= "HE\xad und LLO WORLD"
    
    res2 = dp._regex_normalize_text_ger(text2)
    res3 = dp._regex_normalize_text_ger(text3)
    res4 = dp._regex_normalize_text_ger(text4)
    
    assert res2 == "HELLO"
    assert res3 == "HELLO"
    assert res4 == "HE\xad und LLO WORLD"

def test_remove_processed_data_should_remove_all(dp,out_dirs) -> None:
    chunks, docs = out_dirs
    chunks = (chunks / "test1.jsonl")
    chunks.touch()
    docs = (docs / "test1.json")
    docs.touch()
    dp._remove_processed_data()
    
    assert False == chunks.exists()
    assert False == docs.exists()
    
def test_remove_processed_data_should_remove_specific_files(dp, out_dirs)->None:
    
    chunks, docs = out_dirs
    chunks = (chunks / "test1.jsonl")
    chunks.touch()
    docs = (docs / "test1.json")
    docs.touch()
    dp._remove_processed_data(stem="test1")
    
    assert False == chunks.exists()
    assert False == docs.exists()

def test_manifest_load_should_return_empty_dict_manifest_empty(dp,mock_settings,out_dirs) -> None:
    
    chunks, docs = out_dirs
    chunks = (chunks / "test1.jsonl")
    chunks.touch()
    docs = (docs / "test1.json")
    docs.touch()
    
    d:dict = dp._manifest_load()
    assert isinstance(d, dict)
    assert False == chunks.exists()
    assert False == docs.exists()
    
def test_manifest_load_should_return_manifest(dp,mock_settings, out_dirs) -> None:
    doc_meta = Doc_Hashes(
        source_name="test",
        doc_id= "test",
        tokenizer="test",
        max_tokens="100",
        ceiling=100,
        threshold= 100,
        normalize_text= "test",
        total_hash="test",
    )
    manifest: dict[str, Doc_Hashes] = {"test_doc": doc_meta}
    manifest_path = Path(settings.DOC_MANIFEST_OUT_DIR / "manifest.json")
    manifest_path.write_bytes(MANIFEST_ADAPTER.dump_json(manifest, indent=2))
    
    d = dp._manifest_load()
    
    key,value = list(d.items())[0]
    
    assert key == "test_doc"
    assert value.source_name == "test"
    assert value.doc_id == "test"
    assert value.tokenizer == "test"
    assert value.max_tokens == "100"
    assert value.ceiling == 100
    assert value.threshold == 100
    assert value.normalize_text == "test"
    assert value.total_hash == "test"

def test_manifest_cleaner_should_return_non_empty_manifest(dp,mock_settings,out_dirs):
    #manifest_file = settings.DOC_MANIFEST_OUT_DIR / "manifest.json"
    
    doc_meta = Doc_Hashes(
        source_name="test",
        doc_id= "test",
        tokenizer="test",
        max_tokens="100",
        ceiling=100,
        threshold= 100,
        normalize_text= "test",
        total_hash="test",
    )
    doc_meta2 = Doc_Hashes(
            source_name="test2",
            doc_id= "test2",
            tokenizer="test2",
            max_tokens="100",
            ceiling=100,
            threshold= 100,
            normalize_text= "test2",
            total_hash="test2",
        )
    manifest: dict[str, Doc_Hashes] = {"test_doc": doc_meta,
                                       "test2_doc": doc_meta2}
    
    dp._manifest_cleaner(manifest=manifest, stem="test2_doc")
    
    new_manifest = dp._manifest_load()
    
    assert len(new_manifest.items()) == 1
    assert "test_doc" in new_manifest
    assert "test2_doc" not in new_manifest
    assert type(new_manifest["test_doc"]) == Doc_Hashes
    assert new_manifest["test_doc"].source_name == "test"
    assert new_manifest["test_doc"].doc_id == "test"
    assert new_manifest["test_doc"].tokenizer == "test"
    assert new_manifest["test_doc"].max_tokens == "100"
    assert new_manifest["test_doc"].ceiling == 100
    assert new_manifest["test_doc"].threshold == 100
    assert new_manifest["test_doc"].normalize_text == "test"
    assert new_manifest["test_doc"].total_hash == "test"