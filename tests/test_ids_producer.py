import os

import imas
import numpy as np
import pytest

from imas_streams import StreamingIDSProducer

# Data Dictionary version to use for tests
DD_VERSION = os.getenv("IMAS_VERSION", "4.0.0")


@pytest.fixture
def magnetics():
    ids = imas.IDSFactory(DD_VERSION).new("magnetics")

    ids.ids_properties.homogeneous_time = imas.ids_defs.IDS_TIME_MODE_HOMOGENEOUS
    ids.time = [0.0]

    ids.flux_loop.resize(5)
    for i, loop in enumerate(ids.flux_loop):
        loop.name = f"flux_loop_{i}"
        loop.position.resize(1)
        loop.position[0].r = i / 2
        loop.position[0].z = i / 2
        loop.flux.data = [10.0 + i]
        loop.voltage.data = [i / 10]
    return ids


def test_create_producer(magnetics):
    producer = StreamingIDSProducer(magnetics)
    metadata = producer.metadata

    assert metadata.ids_name == "magnetics"
    assert metadata.data_dictionary_version == DD_VERSION
    assert len(metadata.dynamic_data) == 11  # flux and voltage for 5 flux loops; time

    for i, loop in enumerate(metadata.static_data.flux_loop):
        assert loop.name == f"flux_loop_{i}"
        assert loop.position[0].r == i / 2
        assert loop.position[0].z == i / 2
        # Check that the dynamic data is removed from the static IDS
        assert not loop.flux.data.has_value
        assert not loop.voltage.data.has_value


def test_producer_static_paths(magnetics):
    producer = StreamingIDSProducer(magnetics, static_paths=["flux_loop/flux/data"])
    metadata = producer.metadata

    assert metadata.ids_name == "magnetics"
    assert metadata.data_dictionary_version == DD_VERSION
    assert len(metadata.dynamic_data) == 6  # voltage for 5 flux loops; time

    for i, loop in enumerate(metadata.static_data.flux_loop):
        assert loop.name == f"flux_loop_{i}"
        assert loop.position[0].r == i / 2
        assert loop.position[0].z == i / 2
        # Flux is assumed static data now
        assert np.array_equal(loop.flux.data, [10.0 + i])
        # Check that the dynamic data is removed from the static IDS
        assert not loop.voltage.data.has_value


def test_producer_stream(magnetics):
    producer = StreamingIDSProducer(magnetics)
    assert [data.path for data in producer.metadata.dynamic_data] == [
        "time",
        "flux_loop[0]/flux/data",
        "flux_loop[0]/voltage/data",
        "flux_loop[1]/flux/data",
        "flux_loop[1]/voltage/data",
        "flux_loop[2]/flux/data",
        "flux_loop[2]/voltage/data",
        "flux_loop[3]/flux/data",
        "flux_loop[3]/voltage/data",
        "flux_loop[4]/flux/data",
        "flux_loop[4]/voltage/data",
    ]

    data = producer.create_message(magnetics)
    arr = np.frombuffer(data, dtype=float)
    assert arr[0] == 0.0  # time
    assert np.array_equal(arr[1::2], [10.0 + i for i in range(5)])  # flux/data
    assert np.array_equal(arr[2::2], [i / 10 for i in range(5)])  # voltage/data
