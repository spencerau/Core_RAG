import os
from tika import parser
from pathlib import Path
import tiktoken

TIKA_SERVER_ENDPOINT = os.environ.get(
    "TIKA_SERVER_ENDPOINT",
    "http://tika:9998" if os.environ.get("DOCKER_ENV") else "http://localhost:9998"
)
parser.ServerEndpoint = TIKA_SERVER_ENDPOINT


def extract_content(file_path: str):
    parsed = parser.from_file(file_path, serverEndpoint=TIKA_SERVER_ENDPOINT)
    text = parsed.get('content', '')
    metadata = parsed.get('metadata', {})
    return text, metadata


def extract_content_from_bytes(file_bytes: bytes, file_name: str = "file"):
    parsed = parser.from_buffer(file_bytes, file_name, serverEndpoint=TIKA_SERVER_ENDPOINT)
    text = parsed.get('content', '')
    metadata = parsed.get('metadata', {})
    return text, metadata


def extract_chunks_and_metadata(file_path: str, user_metadata=None, chunk_size=400, chunk_overlap=80):
    parsed = parser.from_file(file_path, serverEndpoint=TIKA_SERVER_ENDPOINT)
    text = parsed.get('content', '')
    tika_metadata = parsed.get('metadata', {})
    payload_metadata = user_metadata or {}
    if tika_metadata:
        payload_metadata = {**tika_metadata, **payload_metadata}
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    stride = chunk_size - chunk_overlap
    chunks = [
        enc.decode(tokens[i : i + chunk_size])
        for i in range(0, len(tokens), stride)
        if tokens[i : i + chunk_size]
    ]
    return [(chunk, payload_metadata) for chunk in chunks]
