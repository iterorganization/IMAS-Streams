import itertools
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from imas.util import to_xarray

from imas_streams import StreamingIMASMetadata

if TYPE_CHECKING:
    import xarray


_index_pattern = re.compile(r"\[(\d+)\]")


def path_to_xarray_name_and_indices(path: str) -> tuple[str, tuple[int, ...]]:
    """Convert the IMAS DD path to its tensorized xarray name and corresponding indices.

    Examples:
        >>> path_to_xarray_name_and_indices("time")
        ('time', ())
        >>> path_to_xarray_name_and_indices("profiles_1d[0]/grid/rho_tor_norm")
        ('profiles_1d.grid.rho_tor_norm', (0,))
        >>> path_to_xarray_name_and_indices("profiles_1d[0]/ion[2]/temperature")
        ('profiles_1d.ion.temperature', (0, 2))
    """
    indices = tuple(int(match.group(1)) for match in _index_pattern.finditer(path))
    path = _index_pattern.sub("", path).replace("/", ".")
    return path, indices


def np_address_of(arr: np.ndarray) -> int:
    """Return the memory address of the first item in the provided numpy array."""
    return arr.__array_interface__["data"][0]


def offset_in_array(array: np.ndarray, index: Iterable[int]) -> int:
    """Calculate the offset (in bytes) of the provided index in the array.

    Examples:
        >>> array = np.arange(15, dtype=float).reshape(3, 5)
        >>> offset_in_array(array, (0, 0))  # First item is stored at offset 0
        0
        >>> offset_in_array(array, (0, 1))  # Second item is offset by 8 bytes
        8
        >>> offset_in_array(array, (1, 0))  # Second row is offset by 5*8 bytes
        40
    """
    return sum(i * stride for i, stride in zip(index, array.strides, strict=False))


class StreamingXArrayConsumer:
    """Consumer of streaming IMAS data which outputs xarray.Datasets.

    This streaming IMAS data consumer updates an xarray.Dataset for each time slice.

    Example:
        .. code-block:: python

            # Create metadata (from JSON)
            metadata = StreamingIMASMetadata.model_validate_json(json_metadata)
            # Create reader
            reader = StreamingXArrayConsumer(metadata)

            # Consume dynamic data
            for dynamic_data in dynamic_data_stream:
                ds = reader.process_message(dynamic_data)
                # Use Dataset
                ...
    """

    def __init__(self, metadata: StreamingIMASMetadata) -> None:
        """Consumer of streaming IMAS data which outputs xarray.Datasets.

        Args:
            metadata: Metadata of the IMAS data stream.
        """
        self._metadata = metadata
        ids = metadata.static_data
        # Add entries for dynamic data in the IDS, so the IMAS-Python to_xarray will
        # create the corresponding xarray.DataArrays for us
        for dyndata in metadata.dynamic_data:
            ids[dyndata.path].value = np.zeros(dyndata.shape)
        self._dataset = to_xarray(ids)
        # pandas is optional (through IMAS-Python), so import locally
        from pandas import Index

        # Setup array view buffer
        buffersize = 0
        tensorized_paths = []
        for dyndata in metadata.dynamic_data:
            assert dyndata.data_type == "f64"
            path = path_to_xarray_name_and_indices(dyndata.path)[0]
            if path not in tensorized_paths:
                tensorized_paths.append(path)
                buffersize += self._dataset[path].size
        dtype = "<f8"  # little-endian IEEE-754 64-bits floating point number
        self._tensor_buffer = np.ndarray(buffersize, dtype=dtype)
        readonly_view = memoryview(self._tensor_buffer).toreadonly()

        # Setup array views
        tensor_idx = 0
        to_update = {}
        tensorviews = {}
        for path in tensorized_paths:
            xrda = self._dataset[path]
            # Fill tensor buffer with initial values of data array
            size = xrda.size
            self._tensor_buffer[tensor_idx : tensor_idx + size] = xrda.data.flatten()
            # And put a readonly view of the tensor buffer back
            buffer = readonly_view[tensor_idx : tensor_idx + size]
            tensorview = np.frombuffer(buffer, dtype=dtype).reshape(xrda.shape)
            tensorviews[path] = tensorview
            if path in self._dataset.indexes:
                # Prevent xarray from creating a copy of the data:
                tensorview = Index(tensorview, copy=False)
            to_update[path] = (xrda.dims, tensorview)
            tensor_idx += size
        self._dataset = self._dataset.assign(to_update)
        # Check that all data arrays are indeed views of our tensor buffer:
        for path, tensorview in tensorviews.items():
            assert np_address_of(self._dataset[path].data) == np_address_of(tensorview)

        # Set up the index array for writing received messages into the tensor buffer:
        self._index_array = np.zeros(metadata.nbytes // 8, dtype=int)
        idx = 0
        for dyndata in metadata.dynamic_data:
            path, indices = path_to_xarray_name_and_indices(dyndata.path)
            # First check if this works before attempting to speed up
            array = self._dataset[path].data
            base_address = np_address_of(array) + offset_in_array(array, indices)
            subarray = array[indices]
            for index in itertools.product(*[range(i) for i in dyndata.shape]):
                self._index_array[idx] = base_address + offset_in_array(subarray, index)
                idx += 1
        self._index_array -= np_address_of(self._tensor_buffer)
        self._index_array //= 8  # go from bytes to indices in the numpy array

        # Message buffer and non-tensorized array view
        self._msg_buffer = memoryview(bytearray(metadata.nbytes))
        self._array_view = np.frombuffer(self._msg_buffer, dtype=dtype)

    def process_message(self, data: bytes | bytearray) -> "xarray.Dataset":
        """Process a dynamic data message and return the resulting xarray Dataset.

        Note that for efficiency we return the same dataset with each call. You should
        not modify the dataset in-place, or future calls to this method may not work
        correctly.

        Args:
            data: Binary data corresponding to one time slice of dynamic data.
        """
        if len(data) != len(self._msg_buffer):
            raise ValueError(
                f"Unexpected size of data: {len(data)}. "
                f"Was expecting {len(self._msg_buffer)}."
            )
        # Copy data to internal buffer, then write into the tensor view:
        self._msg_buffer[:] = data
        self._tensor_buffer[self._index_array] = self._array_view
        return self._dataset

    def finalize(self) -> None:
        """Indicate that the final message is received and return any remaining data."""
        return None  # No data remaining
