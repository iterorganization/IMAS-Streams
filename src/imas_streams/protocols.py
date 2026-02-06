"""Protocols"""

from typing import Any, Protocol

from imas_streams.metadata import StreamingIMASMetadata


class StreamConsumer(Protocol):
    def __init__(self, metadata: StreamingIMASMetadata, **kwargs) -> None: ...
    def process_message(self, data: bytes | bytearray) -> Any: ...
