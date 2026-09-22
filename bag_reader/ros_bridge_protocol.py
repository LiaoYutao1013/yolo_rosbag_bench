from __future__ import annotations

import json
import struct
from typing import Any, BinaryIO


def encode_message(message_type: str, payload: bytes = b"", **fields: Any) -> bytes:
    header = {"type": message_type, **fields, "payload_len": len(payload)}
    header_bytes = json.dumps(header, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(header_bytes)) + header_bytes + payload


def read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("ROS bridge stream closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode_message(stream: BinaryIO) -> tuple[dict[str, Any], bytes]:
    header_len = struct.unpack(">I", read_exact(stream, 4))[0]
    if header_len <= 0 or header_len > 16 * 1024 * 1024:
        raise ValueError(f"Invalid ROS bridge header length: {header_len}")
    header_bytes = read_exact(stream, header_len)
    header = json.loads(header_bytes.decode("utf-8"))
    payload_len = int(header.get("payload_len", 0))
    if payload_len < 0 or payload_len > 256 * 1024 * 1024:
        raise ValueError(f"Invalid ROS bridge payload length: {payload_len}")
    payload = read_exact(stream, payload_len) if payload_len else b""
    return header, payload
