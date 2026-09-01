import pytest
from backend.doc_processing import Doc_Processing
from backend.config import Chunking_Constants
from backend.chunk_schema import Chunk
@pytest.fixture
def cts():
 return Chunking_Constants()

@pytest.fixture
def dp(cts):
    return Doc_Processing(cts, dev_mode=True)

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

    
    
    