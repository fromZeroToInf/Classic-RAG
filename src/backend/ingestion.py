import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"  # windows issue incompatibility
import json
import pprint

from docling.document_converter import DocumentConverter

from backend.config import settings


def read_documents_and_save():
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
