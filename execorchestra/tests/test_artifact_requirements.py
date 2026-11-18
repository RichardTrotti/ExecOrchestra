"""Regression coverage for query artifact inference."""

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Decode the response and return JSON", ["json"]),
        ("Encode the response as JSON", ["json"]),
        ("Decode the response", []),
        ("Discuss coding practices in Markdown", ["markdown"]),
        ("Summarize the codebase in Markdown", ["markdown"]),
        ("Deliver the source code", ["code"]),
        ("Deliver `CODE` in Markdown", ["code", "markdown"]),
        ("Write the source_code.py file", ["code"]),
        ("\u8bf7\u8f93\u51fa\u6e90\u7801", ["code"]),
        ("\u8bf7\u8f93\u51fa\u4ee3\u7801", ["code"]),
        ("\u8bf7\u8f93\u51facode\u3002", ["code"]),
        ("Write the result to report.md", ["markdown"]),
    ],
)
def test_infer_output_artifacts(query, expected, monkeypatch):
    template = Path(__file__).resolve().parents[1] / "configs" / "retrieval_policy.example.json"
    monkeypatch.setenv("SGAR_RETRIEVAL_POLICY_PATH", str(template))
    monkeypatch.setenv("SGAR_SEALED_LOCAL_VALIDATION", "1")

    from retrieve import infer_hard_requirements

    assert infer_hard_requirements(query)["output_artifacts"] == expected
