from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, value: str) -> None:
    atomic_write_bytes(path, value.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    lines = (json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records)
    atomic_write_text(path, "\n".join(lines) + "\n")


def atomic_save_safetensors(path: Path, tensors: dict[str, Any]) -> None:
    """Serialize a tensor mapping and publish it with the run's atomic-write contract."""
    from safetensors.torch import save

    atomic_write_bytes(path, save({key: value.contiguous() for key, value in tensors.items()}))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(*parts: object) -> bytes:
    value = ":".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(value).digest()


def stable_mod(value: str, modulus: int, namespace: str = "worldscene-h3-v1") -> int:
    if modulus <= 0:
        raise ValueError("modulus must be positive.")
    return int.from_bytes(stable_digest(namespace, value)[:8], "big") % modulus
