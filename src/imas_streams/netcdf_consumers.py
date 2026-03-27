import logging
from pathlib import Path

from imas_streams import StreamingIMASMetadata
from imas_streams.xarray_consumers import StreamingXArrayConsumer

logger = logging.getLogger(__name__)
try:
    import netCDF4
    import xarray
except ImportError:
    logger.error("Optional dependency 'netCDF4' or 'xarray' is not installed.")
    raise


class NetCDFConsumer:
    """Consumer of streaming IMAS data which stores the data in a netCDF file.

    Note: writes to the filesystem are batched for better performance. The first data
    will be stored once the first batch is complete.

    Example:
        .. code-block:: python

            # Create metadata (from JSON)
            metadata = StreamingIMASMetadata.model_validate_json(json_metadata)
            # Create reader
            reader = NetCDFConsumer(metadata, filename="output.nc", batch_size=1024)

            # Consume dynamic data
            for dynamic_data in dynamic_data_stream:
                reader.process_message(dynamic_data)
            reader.finalize()
    """

    def __init__(
        self, metadata: StreamingIMASMetadata, *, filename: Path, batch_size: int = 1024
    ):
        self._metadata = metadata
        self._groupname = f"{metadata.ids_name}/0"
        self._filename = filename
        self._batch_size = batch_size

        # Touch the file so we know that it exists
        try:
            self._filename.touch(exist_ok=False)
        except FileExistsError as exc:
            exc.add_note(
                "NetCDFConsumer will not overwrite existing files. Please rename or "
                f"remove {filename} and try again."
            )
            raise
        self._netcdf_file = None
        self._time_variables = []

        # Let the xarray consumer handle all buffering and tensorization:
        self._xarray_consumer = StreamingXArrayConsumer(metadata, batch_size=batch_size)

    def _store_data(self, ds: xarray.Dataset) -> None:
        """Store data to netCDF file"""
        if self._netcdf_file is None:
            # Check which variables are time-dependent
            for varname in ds.variables:
                if "time" in ds[varname].dims:
                    self._time_variables.append(varname)

            # Compress dynamic data
            encoding = {
                name: {"compression": "zlib", "complevel": 1}
                for name in self._time_variables
                if ds[name].dtype.kind in "if"
            }

            # Use xarray API to create the netCDF file
            ds.to_netcdf(
                self._filename,
                "w",
                format="NETCDF4",
                engine="netcdf4",
                group=self._groupname,
                auto_complex=True,
                unlimited_dims=["time"],
                encoding=encoding,
            )
            self._netcdf_file = netCDF4.Dataset(self._filename, "r+", auto_complex=True)
            self._netcdf_file.set_fill_off()  # Don't fill on resize
            # Set mandatory global metadata attributes
            self._netcdf_file.Conventions = "IMAS"
            self._netcdf_file.data_dictionary_version = (
                self._metadata.data_dictionary_version
            )

        else:
            # Add time slices to netCDF file
            group = self._netcdf_file[self._groupname]
            timeslice = slice(len(group.dimensions["time"]), None)
            allslice = slice(None)
            for varname in self._time_variables:
                xrvar = ds[varname]
                index = tuple(
                    timeslice if dim == "time" else allslice for dim in xrvar.dims
                )
                group[varname][index] = xrvar.data

    def process_message(self, data: bytes | bytearray) -> None:
        """Process a dynamic data message and store it (when a batch is full)."""
        result = self._xarray_consumer.process_message(data)
        if result is not None:
            self._store_data(result)

    def finalize(self) -> None:
        """Indicate that the final message is received and store any remaining data."""
        result = self._xarray_consumer.finalize()
        if result is not None:
            self._store_data(result)
        if self._netcdf_file is not None:
            self._netcdf_file.close()
