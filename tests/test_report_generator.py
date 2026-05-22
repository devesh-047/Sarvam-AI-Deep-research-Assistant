"""Tests for evaluation/report_generator.py"""
import json
import os
import tempfile
import pytest
from evaluation.report_generator import generate_report


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_result(id="T001", category="factual", error=None, answer="", citations=None, retrieved_chunks=None):
    return {
        "id": id,
        "category": category,
        "question": "Sample question?",
        "answer": answer,
        "citations": citations or [],
        "retrieved_chunks": retrieved_chunks or [],
        "plan_text": "Research Plan:\n1. Search\n2. Retrieve\n3. Answer",
        "search_queries": ["sample query"],
        "latency_ms": 1234.5,
        "error": error,
    }


def test_generate_report_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "results.jsonl")
        md_path = os.path.join(tmpdir, "report.md")

        records = [
            _make_result("F001", "factual", answer="The answer is X [S1]." * 15,
                         citations=[{"label": "[S1]", "title": "T", "url": "http://x.com", "domain": "x.com"}]),
            _make_result("F002", "factual", answer="", error="Search failed"),
        ]
        _write_jsonl(jsonl_path, records)

        generate_report(jsonl_path, md_path)

        assert os.path.exists(md_path), "Report file should be created"
        with open(md_path) as f:
            content = f.read()
        assert "# Deep Research Assistant" in content
        assert "## Summary" in content
        assert "## Results by Category" in content
        assert "## Per-Question Results" in content


def test_generate_report_contains_category():
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "results.jsonl")
        md_path = os.path.join(tmpdir, "report.md")

        records = [
            _make_result("C001", "comparison", answer="A vs B [S1]." * 20,
                         citations=[{"label": "[S1]", "title": "T", "url": "http://x.com", "domain": "x.com"}]),
        ]
        _write_jsonl(jsonl_path, records)
        generate_report(jsonl_path, md_path)

        with open(md_path) as f:
            content = f.read()
        assert "comparison" in content


def test_generate_report_uncertainty_example():
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "results.jsonl")
        md_path = os.path.join(tmpdir, "report.md")

        records = [
            _make_result("I001", "insufficient_evidence",
                         answer="There is insufficient evidence to answer this [S1]." * 5,
                         citations=[{"label": "[S1]", "title": "T", "url": "http://x.com", "domain": "x.com"}]),
        ]
        _write_jsonl(jsonl_path, records)
        generate_report(jsonl_path, md_path)

        with open(md_path) as f:
            content = f.read()
        assert "Uncertainty" in content
