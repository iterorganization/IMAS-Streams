import copy

import numpy as np
from imas.ids_toplevel import IDSToplevel

from imas_streams.metadata import StreamingIMASMetadata


class StreamingIDSConsumer:
    """Reader of streaming IMAS data that produces IDSs.

    Example:
        .. code-block:: python

            # Create metadata (from JSON):
            metadata = StreamingMetadata.model_validate_json(json_metadata)
            # Create reader
            reader = StreamingIDSReader(metadata)

            # Consume dynamic data
            for dynamic_data in dynamic_data_stream:
                ids = reader.process_message(dynamic_data)
                # Use IDS
                ...
    """

    def __init__(self, metadata: StreamingIMASMetadata):
        self._metadata = metadata
        self._ids = copy.deepcopy(metadata.static_data)
        nbytes = metadata.buffersize
        self._buffer = memoryview(bytearray(nbytes))
        readonly_view = self._buffer.toreadonly()
        dtype = "<f8"  # little-endian IEEE-754 64-bits floating point number

        idx = 0
        for dyndata in metadata.dynamic_data:
            ids_node = self._ids[dyndata.path]
            assert dyndata.data_type == "f64"

            n = np.prod(dyndata.shape, dtype=int)
            dataview = np.frombuffer(readonly_view, dtype=dtype, count=n, offset=idx)
            dataview = dataview.reshape(dyndata.shape)
            ids_node.value = dataview
            # Verify that IMAS-Python keeps the view of our buffer
            assert ids_node.value is dataview

            idx += dataview.nbytes

    def process_message(self, data: bytes, *, return_copy: bool = True) -> IDSToplevel:
        """Process a dynamic data message and return the resulting IDS.

        Args:
            data: Binary data corresponding to one time slice of dynamic data.

        Keyword Args:
            return_copy: By default, a copy of the IDS is returned with each time slice.
                This incurs some overhead, but produces the correct behaviour when
                applications make changes to the IDS and/or want to store previous time
                slices.

                If this argument is set to ``False`` the same IDS object is returned
                with every call to this method, with the underlying data updated. This
                avoids creating a copy with every call, but will produce incorrect
                results in some scenarios:

                - Because the same IDS object is returned with every call to this
                  method, the following does not work::

                    # Process data for the first time slice:
                    ids1 = reader.process_message(data1, return_copy=False)
                    print(ids1.time.value)  # -> [0.]
                    ...

                    # Process data for the second time slice:
                    ids2 = reader.process_message(data2, return_copy=False)
                    print(ids2.time.value)  # -> [0.1]
                    # IDS1 is also updated to the new data2!
                    print(ids1.time.value)  # -> [0.1]

                - You should make any changes to the underlying IDS object, otherwise
                  their values will no longer update. For example::

                    # Process data for the first time slice:
                    ids = reader.process_message(data, return_copy=False)
                    print(ids.time.value)  # -> [0.]

                    # Making changes to the IDS will break the update process:
                    ids.time = [-1.0]
                    print(ids.time.value)  # -> [-1.]

                    # Process data for the second time slice:
                    ids = reader.process_message(data, return_copy=False)
                    # The update mechanism is broken!
                    print(ids.time.value)  # -> [-1.]
                    # Even when requesting a copy:
                    ids = reader.process_message(data, return_copy=True)
                    print(ids.time.value)  # -> [-1.]

        Returns:
            An IDS with both static and dynamic data for the provided time slice.
        """
        if len(data) != len(self._buffer):
            raise ValueError(
                f"Unexpected size of data: {len(data)}. "
                f"Was expecting {len(self._buffer)}."
            )

        # Copy data into our buffer
        self._buffer[:] = data
        if return_copy:
            return copy.deepcopy(self._ids)
        return self._ids
