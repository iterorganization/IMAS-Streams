import base64
from typing import Any, Literal

import imas
import numpy as np
from imas.ids_toplevel import IDSToplevel
from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    model_serializer,
    model_validator,
)

# Size of one element of a certain data type in bytes
_OBJECT_SIZES = {
    "f64": 8,
}


class DynamicData(BaseModel):
    path: str
    """Path inside the IDS, e.g. `flux_loop[0]/flux/data`"""

    shape: tuple[int, ...]
    """Shape of the data in each time slice."""

    # TODO: add other data types, e.g. "f32", "c64", "c128", "i8", "i16", "i32", ...
    data_type: Literal["f64"]
    """String representation of the data type."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def deserialize_dynamicdata(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"path": data, "shape": (), "data_type": "f64"}
        return data

    @model_serializer(mode="wrap")
    def serialize_dynamicdata(self, handler) -> dict[str, str] | str:
        if not self.shape and self.data_type == "f64":
            return self.path
        return handler(self)


class StreamingIMASMetadata(BaseModel):
    """Metadata and static data for IMAS streaming protocol."""

    # Allowed literal options can be expanded in the future
    metadata_version: Literal["0.1"] = "0.1"
    """Version string of the Streaming IMAS metadata"""

    # Actual static and dynamic data:
    data_dictionary_version: str
    """Version of the Data Dictionary of the underlying data."""
    ids_name: str
    """Name of the IDS (e.g. magnetics) for the underlying data"""
    static_data: IDSToplevel  # TODO: make these ND arrays instead of serialized IDSs
    """Static data as an IDS"""
    dynamic_data: list[DynamicData]
    """List of Dynamic Data that is sent for every time slice."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @field_serializer("static_data")
    def serialize_static_data(self, static_data: IDSToplevel, _info) -> str:
        return base64.b64encode(static_data.serialize()).decode()

    @model_validator(mode="before")
    @classmethod
    def deserialize_static_data(cls, data: Any) -> Any:
        """Deserialize the static data IDS when constructing metadata from JSON."""
        if isinstance(data, dict):
            ids_name = data.get("ids_name", None)
            static_data = data.get("static_data", None)
            data_dictionary_version = data.get("data_dictionary_version", None)
            if (
                isinstance(static_data, str)
                and ids_name is not None
                and data_dictionary_version is not None
            ):
                # Static data should be base64 encoded, serialized IDS. Deserialize it:
                try:
                    ids = imas.IDSFactory(data_dictionary_version).new(ids_name)
                except Exception as exc:
                    raise ValueError(
                        f"Could not create {ids_name!r} IDS: {exc}"
                    ) from exc
                try:
                    serialized_ids = base64.b64decode(static_data.encode())
                    ids.deserialize(serialized_ids)
                except Exception as exc:
                    raise ValueError(
                        f"Error while deserializing {ids_name!r} IDS: {exc}"
                    ) from exc
                data["static_data"] = ids
        return data

    @property
    def buffersize(self):
        """Calculate the size of the dynamic data buffer in bytes."""
        objsize = _OBJECT_SIZES
        return sum(
            np.prod(dyndata.shape, dtype=int) * objsize[dyndata.data_type]
            for dyndata in self.dynamic_data
        )
