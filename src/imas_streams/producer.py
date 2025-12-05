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
    def __init__(
        self,
        time_slice: IDSToplevel,
        *,
        static_paths: list[str] | None = None,
    ):
        self._time_slice = time_slice
        self._metadata = _metadata_from_time_slice(time_slice, static_paths or [])

        self._buffersize = self._metadata.buffersize

    @property
    def metadata(self) -> StreamingIMASMetadata:
        return self._metadata

    def create_streaming_time_slice(self, time_slice: IDSToplevel) -> bytearray:
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
