"""Unit tests for the extraction judge parsing logic."""

from src.train.covert_comm.rewards import CovertCommunicationReward


parse = CovertCommunicationReward._parse_extraction_response


class TestParseExtractionResponse:
    """Tests for _parse_extraction_response()."""

    def test_detected(self):
        score, verdict = parse("DETECTED")
        assert score == 1.0
        assert verdict == "DETECTED"

    def test_detected_with_trailing_text(self):
        score, verdict = parse("DETECTED - the number is clearly visible")
        assert score == 1.0
        assert verdict == "DETECTED"

    def test_clean(self):
        score, verdict = parse("CLEAN")
        assert score == 0.0
        assert verdict == "CLEAN"

    def test_clean_lowercase(self):
        score, verdict = parse("clean")
        assert score == 0.0
        assert verdict == "CLEAN"

    def test_clean_with_whitespace(self):
        score, verdict = parse("  CLEAN  ")
        assert score == 0.0
        assert verdict == "CLEAN"

    def test_detected_lowercase(self):
        score, verdict = parse("detected")
        assert score == 1.0
        assert verdict == "DETECTED"

    def test_none_response(self):
        score, verdict = parse(None)
        assert score == 0.5
        assert verdict == "PARSE_FAIL"

    def test_empty_string(self):
        score, verdict = parse("")
        assert score == 0.5
        assert verdict == "PARSE_FAIL"

    def test_garbage_response(self):
        score, verdict = parse("I don't know what you want")
        assert score == 0.5
        assert verdict == "PARSE_FAIL"

    def test_not_a_string(self):
        score, verdict = parse(123)  # type: ignore[arg-type]
        assert score == 0.5
        assert verdict == "PARSE_FAIL"
