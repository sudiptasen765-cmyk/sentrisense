"""
ml/preprocessing/test_clean_text.py

Run with: pytest ml/preprocessing/test_clean_text.py -v

The negation-preservation tests are the most important ones in this file —
they exist specifically to catch a regression where someone "improves"
cleaning later by adding stopword removal and silently breaks sentiment
meaning.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clean_text import (
    cap_repeated_punctuation,
    clean_for_classical,
    clean_for_transformer,
    normalize_whitespace,
    remove_html,
)


class TestNegationPreservation:
    """These must never fail. If they do, sentiment meaning is being lost."""

    def test_not_good_stays_intact(self):
        result = clean_for_classical("The movie was not good.")
        assert "not" in result
        assert "not good" in result

    def test_never_liked_stays_intact(self):
        result = clean_for_classical("I never liked this actor.")
        assert "never" in result

    def test_not_terrible_stays_intact(self):
        result = clean_for_classical("Surprisingly, not terrible.")
        assert "not terrible" in result

    def test_contraction_negation_stays_intact(self):
        result = clean_for_classical("I wouldn't recommend it.")
        assert "wouldn't" in result or "n't" in result

    def test_transformer_mode_also_preserves_negation(self):
        result = clean_for_transformer("This was not good at all.")
        assert "not good" in result


class TestHtmlRemoval:
    def test_br_tags_removed(self):
        result = remove_html("Great film.<br /><br />Would watch again.")
        assert "<br" not in result
        assert "br />" not in result

    def test_entities_unescaped(self):
        result = remove_html("Tom &amp; Jerry was fun.")
        assert "&amp;" not in result
        assert "&" in result

    def test_generic_tags_removed(self):
        result = remove_html("<p>Some <b>bold</b> review text</p>")
        assert "<" not in result
        assert ">" not in result


class TestWhitespaceNormalization:
    def test_collapses_multiple_spaces(self):
        assert normalize_whitespace("too    many   spaces") == "too many spaces"

    def test_collapses_newlines_and_tabs(self):
        assert normalize_whitespace("line one\n\nline two\t\ttabbed") == "line one line two tabbed"

    def test_trims_leading_trailing(self):
        assert normalize_whitespace("   padded text   ") == "padded text"


class TestRepeatedPunctuation:
    def test_caps_exclamation_runs(self):
        result = cap_repeated_punctuation("amazing!!!!!!!!!")
        assert result == "amazing!!!"

    def test_short_runs_untouched(self):
        result = cap_repeated_punctuation("wait...")
        assert result == "wait..."

    def test_preserves_emphasis_signal(self):
        # We cap length, but the exclamation itself must survive — it's a
        # real sentiment signal, not noise.
        result = cap_repeated_punctuation("terrible!!!!!")
        assert "!" in result


class TestClassicalVsTransformerModes:
    def test_classical_lowercases(self):
        result = clean_for_classical("This Was GREAT")
        assert result == result.lower()

    def test_transformer_preserves_case(self):
        result = clean_for_transformer("This Was GREAT")
        assert "GREAT" in result  # case must survive for the subword tokenizer

    def test_both_modes_strip_html(self):
        raw = "Good movie.<br />Would recommend."
        assert "<br" not in clean_for_classical(raw)
        assert "<br" not in clean_for_transformer(raw)


class TestEdgeCases:
    def test_empty_string(self):
        assert clean_for_classical("") == ""
        assert clean_for_transformer("") == ""

    def test_non_string_input_returns_empty(self):
        assert clean_for_classical(None) == ""
        assert clean_for_classical(123) == ""

    def test_whitespace_only_string(self):
        assert clean_for_classical("   \n\t  ") == ""
