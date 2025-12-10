import os

import imas
import pytest
from pydantic import ValidationError

from imas_streams.metadata import DynamicData, StreamingIMASMetadata

DD_VERSION = os.getenv("IMAS_VERSION", "4.0.0")


def test_dynamicdata():
    # Constructing in Python
    data = DynamicData(path="test", shape=(1,), data_type="f64")

    assert data.path == "test"
    assert data.shape == (1,)
    assert data.data_type == "f64"

    # Invalid data type
    with pytest.raises(ValidationError):
        DynamicData(path="test", shape=(1,), data_type="x64")
    # Invalid shape
    with pytest.raises(ValidationError):
        DynamicData(path="test", shape="1", data_type="x64")


@pytest.fixture
def magnetics_metadata():
    ids = imas.IDSFactory(DD_VERSION).new("magnetics")

    ids.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    ids.time = [0.0]

    ids.flux_loop.resize(5)
    for i, loop in enumerate(ids.flux_loop):
        loop.name = f"flux_loop_{i}"
        loop.position.resize(1)
        loop.position[0].r = i / 2
        loop.position[0].z = i / 2

    return StreamingIMASMetadata(
        data_dictionary_version=DD_VERSION,
        ids_name="magnetics",
        static_data=ids,
        dynamic_data=[
            DynamicData(path="time", shape=(1,), data_type="f64"),
            DynamicData(path="flux_loop[0]/flux/data", shape=(1,), data_type="f64"),
            DynamicData(path="flux_loop[1]/flux/data", shape=(1,), data_type="f64"),
            DynamicData(path="flux_loop[2]/flux/data", shape=(1,), data_type="f64"),
            DynamicData(path="flux_loop[3]/flux/data", shape=(1,), data_type="f64"),
            DynamicData(path="flux_loop[4]/flux/data", shape=(1,), data_type="f64"),
            DynamicData(path="flux_loop[0]/voltage/data", shape=(1,), data_type="f64"),
        ],
    )


def test_metadata(magnetics_metadata):
    assert magnetics_metadata.nbytes == 7 * 8

    data = vars(magnetics_metadata)
    # Must have dynamic data
    data["dynamic_data"] = []
    with pytest.raises(ValidationError, match=r"missing"):
        StreamingIMASMetadata(**data)
    # First dynamic data must be time
    data["dynamic_data"] = [DynamicData(path="test", shape=(1,), data_type="f64")]
    with pytest.raises(ValidationError, match=r"time"):
        StreamingIMASMetadata(**data)
    # Dynamic data paths must be unique
    data["dynamic_data"] = [DynamicData(path="time", shape=(1,), data_type="f64")] * 2
    with pytest.raises(ValidationError, match=r"paths must be unique"):
        StreamingIMASMetadata(**data)


def test_metadata_serialization(magnetics_metadata):
    json = magnetics_metadata.model_dump_json()
    deserialized = StreamingIMASMetadata.model_validate_json(json)
    assert deserialized == magnetics_metadata
