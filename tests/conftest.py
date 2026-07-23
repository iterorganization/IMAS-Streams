import os

import imas
import pytest
from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS

from imas_streams import StreamingIMASMetadata
from imas_streams.metadata import DynamicData

DD_VERSION = os.getenv("IMAS_VERSION", "4.0.0")


@pytest.fixture
def magnetics_metadata():
    ids = imas.IDSFactory(DD_VERSION).new("magnetics")

    ids.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
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


@pytest.fixture
def kafka_host():
    value = os.getenv("KAFKA_HOST")
    if not value:
        pytest.fail("Cannot connect to Kafka server: KAFKA_HOST not set.")
    return value
