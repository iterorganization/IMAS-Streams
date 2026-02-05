from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from imas_streams import StreamingIMASMetadata


class MessageProcessor:
    """Common logic for building data arrays from streaming IMAS messages"""

    def __init__(self, metadata: "StreamingIMASMetadata"):
        self._metadata = metadata
        self._msg_buffer = memoryview(bytearray(metadata.nbytes))
        readonly_view = self._msg_buffer.toreadonly()
        dtype = "<f8"  # little-endian IEEE-754 64-bits floating point number
        self._array_view = np.frombuffer(readonly_view, dtype=dtype)

    def get_array_views(self) -> list[np.ndarray]:
        """Get list of readonly array views of the streamed data.

        Each item in the list corresponds to the entry in metadata.dynamic_data with the
        same index.
        """
        array_views = []
        idx = 0
        for dyndata in self._metadata.dynamic_data:
            assert dyndata.data_type == "f64"
            n = np.prod(dyndata.shape, dtype=int)
            dataview = self._array_view[idx : idx + n].reshape(dyndata.shape)
            array_views.append(dataview)
            idx += n
        return array_views

    def set_data(self, data: bytes | bytearray) -> None:
        """Update array views with data from a new message."""
        if len(data) != len(self._msg_buffer):
            raise ValueError(
                f"Unexpected size of data: {len(data)}. "
                f"Was expecting {len(self._msg_buffer)}."
            )
        self._msg_buffer[:] = data
