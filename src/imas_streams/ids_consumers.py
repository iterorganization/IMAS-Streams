import copy

import numpy as np
from imas.ids_toplevel import IDSToplevel

from imas_streams.imas_utils import get_dynamic_aos_ancestor, get_path_from_aos
from imas_streams.metadata import StreamingIMASMetadata


class MessageProcessor:
    """Logic for building data arrays from streaming IMAS messages"""

    def __init__(self, metadata: StreamingIMASMetadata):
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


class StreamingIDSConsumer:
    """Consumer of streaming IMAS data which outputs IDSs.

    This streaming IMAS data consumer produces an IDS for each time slice.

    Example:
        .. code-block:: python

            # Create metadata (from JSON):
            metadata = StreamingIMASMetadata.model_validate_json(json_metadata)
            # Create reader
            reader = StreamingIDSConsumer(metadata)

            # Consume dynamic data
            for dynamic_data in dynamic_data_stream:
                ids = reader.process_message(dynamic_data)
                # Use IDS
                ...
    """

    def __init__(
        self, metadata: StreamingIMASMetadata, *, return_copy: bool = True
    ) -> None:
        """Consumer of streaming IMAS data which outputs IDSs.

        Args:
            metadata: Metadata of the IMAS data stream.

        Keyword Args:
            return_copy: By default, a copy of the IDS is returned with each time slice.
                This incurs some overhead, but produces the correct behaviour when
                applications make changes to the IDS and/or want to store previous time
                slices.

                If this argument is set to ``False`` the same IDS object is returned
                with every call to this ``process_message``, with the underlying data
                updated. This avoids creating a copy with every call, but will produce
                incorrect results in some scenarios:

                - Because the same IDS object is returned with every call to this
                  method, the following does not work::

                    reader = StreamingIDSConsumer(metadata, return_copy=False)
                    # Process data for the first time slice:
                    ids1 = reader.process_message(data1)
                    print(ids1.time.value)  # -> [0.]
                    ...

                    # Process data for the second time slice:
                    ids2 = reader.process_message(data2)
                    print(ids2.time.value)  # -> [0.1]
                    # IDS1 is also updated to the new data2!
                    print(ids1.time.value)  # -> [0.1]

                - You should not make any changes to the underlying IDS object,
                  otherwise their values will no longer update. For example::

                    reader = StreamingIDSConsumer(metadata, return_copy=False)
                    # Process data for the first time slice:
                    ids = reader.process_message(data)
                    print(ids.time.value)  # -> [0.]

                    # Making changes to the IDS will break the update process:
                    ids.time = [-1.0]
                    print(ids.time.value)  # -> [-1.]

                    # Process data for the second time slice:
                    ids = reader.process_message(data)
                    # The update mechanism is broken!
                    print(ids.time.value)  # -> [-1.]
        """
        self._metadata = metadata
        self._return_copy = return_copy
        self._ids = copy.deepcopy(metadata.static_data)
        self._processor = MessageProcessor(metadata)

        self._scalars = []
        views = self._processor.get_array_views()
        for dyndata, dataview in zip(metadata.dynamic_data, views, strict=True):
            ids_node = self._ids[dyndata.path]
            if ids_node.metadata.ndim == 0:
                self._scalars.append((dyndata, dataview))
            else:
                ids_node.value = dataview
                # Verify that IMAS-Python keeps the view of our buffer
                assert ids_node.value is dataview

    def process_message(self, data: bytes | bytearray) -> IDSToplevel:
        """Process a dynamic data message and return the resulting IDS.

        Args:
            data: Binary data corresponding to one time slice of dynamic data.

        Returns:
            An IDS with both static and dynamic data for the provided time slice.
        """
        self._processor.set_data(data)

        # Copy scalars
        for dyndata, dataview in self._scalars:
            self._ids[dyndata.path] = dataview

        if self._return_copy:
            return copy.deepcopy(self._ids)
        return self._ids


class BatchedIDSConsumer:
    """Consumer of streaming IMAS data which outputs IDSs.

    This streaming IMAS data consumer produces an IDS for every N time slices.

    Example:
        .. code-block:: python

            # Create metadata (from JSON):
            metadata = StreamingIMASMetadata.model_validate_json(json_metadata)
            # Create reader
            reader = BatchedIDSConsumer(metadata, batch_size=100)

            # Consume dynamic data
            for dynamic_data in dynamic_data_stream:
                # process_message returns an IDS after every 100 (=batch_size) messages
                # and None otherwise:
                ids = reader.process_message(dynamic_data)
                if ids is not None:
                    # Use IDS
                    ...
    """

    def __init__(
        self, metadata: StreamingIMASMetadata, batch_size: int, *, return_copy=True
    ) -> None:
        """Consumer of streaming IMAS data which outputs IDSs.

        Args:
            metadata: Metadata of the IMAS data stream.
            batch_size: Number of time slices to batch in each returned IDS.

        Keyword Args:
            return_copy: See the description in StreamingIDSConsumer
        """
        if batch_size < 1:
            raise ValueError(f"Invalid batch size: {batch_size}")

        self._metadata = metadata
        self._batch_size = batch_size
        self._return_copy = return_copy
        self._ids = copy.deepcopy(metadata.static_data)
        self._cur_idx = 0

        self._msg_bytes = metadata.nbytes
        self._buffer = memoryview(bytearray(self._msg_bytes * batch_size))
        readonly_view = self._buffer.toreadonly()
        dtype = "<f8"  # little-endian IEEE-754 64-bits floating point number
        self._array_view = np.frombuffer(readonly_view, dtype=dtype).reshape(
            (batch_size, -1)
        )

        # Setup array views for batched data
        self._scalars = []
        idx = 0
        for dyndata in self._metadata.dynamic_data:
            assert dyndata.data_type == "f64"
            ids_node = self._ids[dyndata.path]
            assert ids_node.metadata.type.is_dynamic
            n = np.prod(dyndata.shape, dtype=int)
            if (
                dyndata.path == "time"
                or ids_node.metadata.ndim
                and ids_node.metadata.coordinates[0].is_time_coordinate
            ):
                # Great! This IDS node is time-dependent by itself, and we can create a
                # single view for it:
                new_shape = (batch_size,) + dyndata.shape[1:]
                dataview = self._array_view[:, idx : idx + n].reshape(new_shape)
                ids_node.value = dataview
                # Verify that IMAS-Python keeps the view of our buffer
                assert ids_node.value is dataview
            else:
                # This is a dynamic variable inside a time-dependent AoS: find that aos
                aos = get_dynamic_aos_ancestor(ids_node)
                # First ensure there's an entry for every batch_size time slices:
                if len(aos) != batch_size:
                    assert len(aos) == 1
                    aos.resize(batch_size, keep=True)
                    for i in range(1, batch_size):
                        aos[i] = copy.deepcopy(aos[0])
                path_from_aos = get_path_from_aos(dyndata.path, aos)
                if ids_node.metadata.ndim == 0:
                    # This is a scalar node
                    self._scalars.append((aos, path_from_aos, idx))
                else:
                    # Loop over all time slices and create views:
                    for i in range(batch_size):
                        dataview = self._array_view[i, idx : idx + n]
                        aos[i][path_from_aos].value = dataview
                        # Verify that IMAS-Python keeps the view of our buffer
                        assert aos[i][path_from_aos].value is dataview

            idx += n

    def process_message(self, data: bytes | bytearray) -> IDSToplevel | None:
        """Process a single streaming IMAS message.

        This method returns None until a full batch is completed. Once ``batch_size``
        messages are processed a single IDSToplevel is returned, which contains all data
        from the ``batch_size`` messages.
        """
        nbytes = self._msg_bytes
        if len(data) != nbytes:
            raise ValueError(
                f"Unexpected size of data: {len(data)}. Was expecting {nbytes}."
            )
        # Update buffer
        self._buffer[self._cur_idx * nbytes : (self._cur_idx + 1) * nbytes] = data
        # Set scalar values
        for aos, path_from_aos, idx in self._scalars:
            aos[self._cur_idx][path_from_aos] = self._array_view[self._cur_idx, idx]

        # Bookkeeping
        self._cur_idx += 1
        if self._cur_idx == self._batch_size:
            # Completed a batch:
            self._cur_idx = 0
            if self._return_copy:
                return copy.deepcopy(self._ids)
            return self._ids
        # Batch is not finished yet
        return None
