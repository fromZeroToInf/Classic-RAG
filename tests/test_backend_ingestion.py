import json
from unittest.mock import MagicMock, patch
from backend.config import settings
from backend.ingestion import read_documents_and_save

def test_read_documents_and_save(tmp_path, monkeypatch):
    
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    
    input_dir.mkdir()
    (input_dir / "dummy.pdf").write_text("test content")
    
    monkeypatch.setattr(settings, "DOCUMENTS_IN_DIR", input_dir)
    monkeypatch.setattr(settings, "DOCUMENTS_OUT_DIR", output_dir)
    
    dummy_doc = MagicMock()
    dummy_doc.export_to_dict.return_value = {"texts": ["dummy"]}
    
    dummy_result = MagicMock()
    dummy_result.status = "SUCCESS"
    dummy_result.document = dummy_doc
    
    with patch("backend.ingestion.DocumentConverter") as MockConverter:
        MockConverter.return_value.convert.return_value = dummy_result
        read_documents_and_save()
    
    output_file = output_dir / "dummy.json"
    
    assert output_file.exists()
    
    data = json.loads(output_file.read_text(encoding="utf-8"))
    assert data == {"texts": ["dummy"]}
    
    