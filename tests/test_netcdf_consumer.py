import imas
import numpy as np
import pytest

from imas_streams.netcdf_consumers import NetCDFConsumer


@pytest.mark.parametrize("batch_size", [1, 2, 5, 7, 10, 13, 20])
def test_netcdf_consumer(magnetics_metadata, tmp_path, batch_size):
    fname = tmp_path / "test.nc"
    reader = NetCDFConsumer(magnetics_metadata, filename=fname, batch_size=batch_size)

    # Pretend sending 20 messages
    for i in range(20):
        test_data = np.arange(len(magnetics_metadata.dynamic_data), dtype="<f8") + i
        dataset = reader.process_message(test_data.tobytes())
        # Only expect a result after batch_size items are processed
        assert dataset is None
    reader.finalize()

    # Check that the file is as expected
    with imas.DBEntry(str(fname), "r") as entry:
        ids = entry.get("magnetics")

    assert np.array_equal(ids.time, np.arange(20, dtype=float))
