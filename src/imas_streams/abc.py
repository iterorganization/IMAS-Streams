"""Abstract base classes"""

from typing import Any

from imas_streams.metadata import StreamingIMASMetadata


class StreamConsumer:
    def __init__(self, metadata: StreamingIMASMetadata) -> None: ...
    def process_message(self, data: bytes | bytearray) -> Any: ...
