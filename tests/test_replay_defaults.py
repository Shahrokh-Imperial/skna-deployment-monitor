from pathlib import Path

def test_public_defaults_in_app():
    text=(Path(__file__).parents[1]/"app/app.py").read_text()
    assert "Pre-event baseline duration [s]'" in text
    assert "60.0" in text
    assert "Persistence updates',1,10,2,1" in text
    assert "Use frozen manuscript threshold" not in text
