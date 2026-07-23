# Integration tests for streaming IDSs
import imas.training
import pytest
from imas.ids_defs import CLOSEST_INTERP

from imas_streams import BatchedIDSConsumer, StreamingIDSConsumer, StreamingIDSProducer
from imas_streams.netcdf_consumers import NetCDFConsumer
from imas_streams.xarray_consumers import StreamingXArrayConsumer


def get_training_db_entry():
    try:
        # convert parameter added in IMAS-Python 2.3
        return imas.training.get_training_db_entry(convert=True)
    except TypeError:
        return imas.training.get_training_db_entry()


@pytest.fixture(scope="module")
def testdb():
    with get_training_db_entry() as entry:
        yield entry


# INT and STR are not supported for streaming, but it is actually static data:
cp_static_paths = [
    "profiles_1d/ion/element/z_n",
    "profiles_1d/ion/element/atoms_n",
    "profiles_1d/ion/name",
    "profiles_1d/ion/neutral_index",
    "profiles_1d/ion/multiple_states_flag",
    "profiles_1d/neutral/element/z_n",
    "profiles_1d/neutral/element/atoms_n",
    "profiles_1d/neutral/name",
    "profiles_1d/neutral/ion_index",
    "profiles_1d/neutral/multiple_states_flag",
]


def test_stream_core_profiles(testdb):
    ids_name = "core_profiles"
    times = testdb.get(ids_name, lazy=True).time.value
    first_slice = testdb.get_slice(ids_name, times[0], CLOSEST_INTERP)
    producer = StreamingIDSProducer(first_slice, static_paths=cp_static_paths)
    consumer = StreamingIDSConsumer(producer.metadata, return_copy=False)

    for t in times:
        time_slice = testdb.get_slice(ids_name, t, CLOSEST_INTERP)
        data = producer.create_message(time_slice)

        deserialized = consumer.process_message(data)
        # Check that the data is identical
        assert list(imas.util.idsdiffgen(time_slice, deserialized)) == []


def test_stream_core_profiles_batched(testdb):
    ids_name = "core_profiles"
    times = testdb.get(ids_name, lazy=True).time.value
    first_slice = testdb.get_slice(ids_name, times[0], CLOSEST_INTERP)
    producer = StreamingIDSProducer(first_slice, static_paths=cp_static_paths)
    consumer = BatchedIDSConsumer(producer.metadata, len(times), return_copy=False)

    for i, t in enumerate(times):
        time_slice = testdb.get_slice(ids_name, t, CLOSEST_INTERP)
        data = producer.create_message(time_slice)

        deserialized = consumer.process_message(data)
        if i != len(times) - 1:
            assert deserialized is None

    # Compare against full IDS
    ids = testdb.get(ids_name)
    # Check that the data is identical
    assert deserialized is not None
    assert list(imas.util.idsdiffgen(ids, deserialized)) == []


def test_stream_core_profiles_xarray(testdb):
    ids_name = "core_profiles"
    times = testdb.get(ids_name, lazy=True).time.value
    first_slice = testdb.get_slice(ids_name, times[0], CLOSEST_INTERP)
    producer = StreamingIDSProducer(first_slice, static_paths=cp_static_paths)
    consumer = StreamingXArrayConsumer(producer.metadata)

    for t in times:
        time_slice = testdb.get_slice(ids_name, t, CLOSEST_INTERP)
        data = producer.create_message(time_slice)

        xrds_orig = imas.util.to_xarray(time_slice)
        xrds_deserialized = consumer.process_message(data)
        # Check that both datasets are identical
        assert xrds_orig.equals(xrds_deserialized)


def test_stream_core_profiles_xarray_batched(testdb):
    ids_name = "core_profiles"
    times = testdb.get(ids_name, lazy=True).time.value
    first_slice = testdb.get_slice(ids_name, times[0], CLOSEST_INTERP)
    producer = StreamingIDSProducer(first_slice, static_paths=cp_static_paths)
    consumer = StreamingXArrayConsumer(producer.metadata, batch_size=len(times))

    for i, t in enumerate(times):
        time_slice = testdb.get_slice(ids_name, t, CLOSEST_INTERP)
        data = producer.create_message(time_slice)

        xrds_deserialized = consumer.process_message(data)
        if i != len(times) - 1:
            assert xrds_deserialized is None

    # Compare against full IDS
    ids = testdb.get(ids_name)
    xrds_orig = imas.util.to_xarray(ids)
    # Check that the data is identical
    assert xrds_deserialized is not None
    assert xrds_orig.equals(xrds_deserialized)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_stream_core_profiles_netcdf(testdb, tmp_path, batch_size):
    ids_name = "core_profiles"
    times = testdb.get(ids_name, lazy=True).time.value
    first_slice = testdb.get_slice(ids_name, times[0], CLOSEST_INTERP)
    producer = StreamingIDSProducer(first_slice, static_paths=cp_static_paths)
    fname = tmp_path / "test.nc"
    consumer = NetCDFConsumer(producer.metadata, filename=fname, batch_size=batch_size)

    for t in times:
        time_slice = testdb.get_slice(ids_name, t, CLOSEST_INTERP)
        data = producer.create_message(time_slice)

        result = consumer.process_message(data)
        assert result is None
    result = consumer.finalize()
    assert result is None

    # Check that the file is as expected
    with imas.DBEntry(str(fname), "r") as entry:
        ids = entry.get(ids_name)

    # Compare against full IDS
    ids_orig = testdb.get(ids_name)
    # N.B. imas.util.to_xarray doesn't include metadata for inhomogeneously sized AoS
    # so we ignore all differences for the size of profiles_1d/ion/state AoS
    diffs = [
        diff
        for diff in imas.util.idsdiffgen(ids, ids_orig)
        if diff[0] != "profiles_1d/ion/state"
    ]
    assert diffs == []
