import json
from pathlib import Path

from semanticflow.dataset import ReviewRecord
from semanticflow.storage.results import ResultStore, dataset_fingerprint


def test_unique_result_directories_and_valid_jsonl(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    _, first, _ = store.create_run("q")
    _, second, _ = store.create_run("q")
    assert first != second
    store.write_jsonl(first / "rows.jsonl", [ReviewRecord("a", "text", {}).to_dict()])
    assert json.loads((first / "rows.jsonl").read_text(encoding="utf-8"))["review_id"] == "a"


def test_dataset_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "file"
    path.write_bytes(b"stable")
    assert dataset_fingerprint(path) == dataset_fingerprint(path)
    assert dataset_fingerprint(path).startswith("sha256:")

