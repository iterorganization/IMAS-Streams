import copy

import imas
import numpy as np
from imas.ids_data_type import IDSDataType
from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS
from imas.ids_toplevel import IDSToplevel

from imas_streams.metadata import DynamicData, StreamingIMASMetadata


def _metadata_from_time_slice(time_slice: IDSToplevel, static_paths: list[str]):
    # -- Data sanity checks --
    # The IDS must use homogeneous time mode
    if time_slice.ids_properties.homogeneous_time != IDS_TIME_MODE_HOMOGENEOUS:
        raise ValueError(
            f"{time_slice} must use homogeneous time, found "
            "ids_properties.homogeneous_time="
            f"{time_slice.ids_properties.homogeneous_time.value} instead"
        )
    # The IDS must contain time-dependent data
    if not hasattr(time_slice, "time"):
        raise ValueError("Cannot stream constant IDSs")
    # The IDS must contain exactly 1 time slice
    if time_slice.time.shape != (1,):
        raise ValueError(
            f"IDS {time_slice} contains {len(time_slice.time)} time slices, "
            "where exactly 1 is expected."
        )
    # All provided static paths must exist in the IDS
    for path in static_paths:
        try:
            time_slice.metadata[path]
        except KeyError:
            raise ValueError(f"Invalid static path provided: {path}") from None
    # The IDS must have valid coordinates and dimensions
    time_slice.validate()

    # -- Split static and dynamic data --
    static = copy.deepcopy(time_slice)
    # Ensure time is first:
    dynamic = [DynamicData(path="time", shape=(1,), data_type="f64")]
    for node in imas.util.tree_iter(time_slice):
        # Trust the DD metadata whether data is dynamic or not
        if not node.metadata.type.is_dynamic:
            continue
        if node.metadata.path_string in static_paths:
            # User overrule: this data won't change over time
            continue

        # Support streaming FLT data only for now
        if node.metadata.data_type is not IDSDataType.FLT:
            raise ValueError(
                f"Streaming {node.metadata.data_type} is not (yet) supported."
            )

        # FIXME: get_full_path() doesn't scale well!
        # Assume for now that we don't stream deeply nested structures (no GGD grids!)
        path = imas.util.get_full_path(node)
        if path != "time":  # time is already in the list
            dynamic.append(DynamicData(path=path, shape=node.shape, data_type="f64"))
        # Remove this node from the static data
        staticparent = imas.util.get_parent(static[path])
        delattr(staticparent, node.metadata.name)

    return StreamingIMASMetadata(
        data_dictionary_version=imas.util.get_data_dictionary_version(time_slice),
        ids_name=time_slice.metadata.name,
        static_data=static,
        dynamic_data=dynamic,
    )


class StreamingIDSProducer:
    """Streaming IMAS data producer, which reads data from IDSs.

    This streaming IMAS data producer reads data from IDSs to produce metadata and a
    data stream in the streaming IMAS format. You can use this producer as follows, see
    also the examples below:

    1.  Create a new StreamingIDSProducer from an IDS. This IDS must use homogeneous
        time (``ids_properties.homogeneous_time ==
        imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS``), contain a single time slice, and
        pass coordinate validation (``ids.validate()``).

        You should fill this IDS with:

        - All static data: all data that is not time-dependent, for example the machine
          description for this IDS. See below explainer on static data.
        - Placeholders for all dynamic data: ensure that every time-dependent quantity
          that you want to stream has a placeholder value in the IDS. The value of the
          placeholder is not used, only the shape of the data (which must remain
          constant during the stream) is important at this stage.

    2.  Extract the streaming metadata and provide this to the consumer.
    3.  Create a message for every time slice by providing an IDS with updated dynamic
        data.

    **Static data**:

    Static data is data that doesn't change during this IMAS data stream. By default it
    encompasses all data that is marked in the `Data Dictionary
    <https://imas-data-dictionary.readthedocs.io/en/latest/coordinates.html#static-constant-and-dynamic-nodes>`__
    as Constant or Static. Data that the Data Dictionary considers Dynamic (i.e.
    time-dependent) can be manually marked as static: for example, your data contains 2D
    equilibrium data. The Data Dictionary indicates that the grid coordinates
    (``time_slice(itime)/profiles_2d(i1)/grid/dim1`` and ``dim2``) are time-dependent.
    However, if you know that these values don't change for this IMAS stream, you can
    mark these as static paths. Doing this will reduce the amount of data sent on every
    time slice, resulting in less communication overhead. See below for an example.

    Examples:
        .. code-block:: python
            :caption: Standard usage

            # Step 1:
            # IDS containing static data and placeholders for dynamic data
            magnetics = imas.IDSFactory().magnetics()
            magnetics.ids_properties.homogeneous_time = 1
            magnetics.time = [0.0]
            magnetics.flux_loop.resize(2)
            magnetics.flux_loop[0].position.resize(1)
            magnetics.flux_loop[0].position[0].r = 2.34
            magnetics.flux_loop[0].position[0].z = 1.12
            magnetics.flux_loop[1].position.resize(1)
            magnetics.flux_loop[1].position[0].r = 2.34
            magnetics.flux_loop[1].position[0].z = -1.08
            # Placeholders for dynamic data
            magnetics.flux_loop[0].flux.data = [0.0]
            magnetics.flux_loop[1].flux.data = [0.0]

            producer = StreamingIDSProducer(magnetics)
            # Step 2:
            metadata = producer.metadata
            # Send metadata to the consumer
            ...

            # Step 3:
            for time in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                # Update dynamic data:
                magnetics.time = [time]
                magnetics.flux_loop[0].flux.data = [1.0 + time]
                magnetics.flux_loop[1].flux.data = [1.0 - time]
                # Create a message
                message = producer.create_message(magnetics)
                # Send the message to the consumer
                ...

        .. code-block:: python
            :caption: Mark additional data as static

            # Get an equilibrium IDS for the first time slice
            equilibrium = calc_equilibrium(t=0, ...)
            # We know that the grid is constant, so we don't want to send this with
            # every time slice
            static_paths = [
                "time_slice/profiles_2d/grid/dim1",
                "time_slice/profiles_2d/grid/dim2",
            ]
            # Step 1: create the producer
            producer = StreamingIDSProducer(equilibrium, static_paths=static_paths)

            # Step 2: Extract metadata and share with consumer
            metadata = producer.metadata
            ...

            # Step 3: Create a message for every time slice
            while True:
                message = producer.create_message(equilibrium)
                ...  # Send to consumer

                t += dt
                if t >= t_stop:
                    break
                equilibrium = calc_equilibrium(t=t, ...)
    """

    def __init__(
        self,
        time_slice: IDSToplevel,
        *,
        static_paths: list[str] | None = None,
    ):
        self._time_slice = time_slice
        self._metadata = _metadata_from_time_slice(time_slice, static_paths or [])

        self._buffersize = self._metadata.nbytes

    @property
    def metadata(self) -> StreamingIMASMetadata:
        return self._metadata

    def create_message(self, time_slice: IDSToplevel) -> bytearray:
        buffer = bytearray(self._buffersize)
        curindex = 0
        for dyndata in self._metadata.dynamic_data:
            node = time_slice[dyndata.path]
            if node.shape != dyndata.shape:
                raise ValueError(
                    f"{dyndata.path} has changed shape, this is not allowed when "
                    f"streaming IMAS data. Initial shape: {dyndata.shape}, "
                    f"current shape: {node.shape}"
                )

            arr: np.ndarray = node.value
            if arr.dtype != "<f8":
                raise ValueError(
                    f"Unexpected data type for {dyndata.path}: {arr.dtype}"
                )
            nbytes = arr.nbytes
            buffer[curindex : curindex + nbytes] = arr.tobytes()
            curindex += nbytes

        if not (curindex == len(buffer) == self._buffersize):
            raise RuntimeError("Internal error: incorrect size of data buffer")
        return buffer
