import logging
import sys

import click
import imas
from imas.ids_defs import CLOSEST_INTERP, IDS_TIME_MODE_HOMOGENEOUS

from imas_streams import BatchedIDSConsumer, StreamingIDSProducer

_PROGRESS_BAR_UPDATE_MINSTEP = 1001


@click.group(invoke_without_command=True, no_args_is_help=True)
@click.version_option()
def main() -> None:
    """Command line utilities for streaming IMAS data."""
    # Disable IMAS-Python log handler (prevent double output for imas log messages)
    imas_logger = logging.getLogger("imas")
    for handler in imas_logger.handlers:
        imas_logger.removeHandler(handler)
    # Set up our own basic log hander, writing messages to sys.stderr
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


@main.command()
@click.argument("imas_uri")
@click.argument("kafka_host")
@click.argument("kafka_topic")
@click.option(
    "--get",
    is_flag=True,
    help="Get full IDS instead of iteratively requesting a time slice with get_slice.",
)
@click.option("-n", default=0, help="Maximum number of time slices to stream")
def imasentry_to_kafka(
    imas_uri: str, kafka_host: str, kafka_topic: str, get: bool, n: int
) -> None:
    """Stream data from an existing IMAS data entry to a Kafka topic.

    The input data must be limited to dynamic floating point data, and array shapes must
    remain constant for all time slices. An error will be displayed when this is not
    adhered to.

    \b
    Arguments:
        IMAS_URI    IMAS URI (including IDS and optionally occurrence) with the data to
                    be streamed. For example: "imas:hdf5?path=./testdata#magnetics".
        KAFKA_HOST  Kafka host and port (aka bootstrap.servers). E.g. 'localhost:9092'.
        KAFKA_TOPIC Name of the kafka topic to stream the data to.
    """
    # Local import: kafka is an optional dependency
    from imas_streams.kafka import KafkaProducer, KafkaSettings

    # Extract IDS/occurrence
    base_uri, _, ids_and_occurrence = imas_uri.partition("#")
    idsname, _, occurrence = ids_and_occurrence.partition(":")
    if not idsname:
        raise click.UsageError(
            f"Invalid IMAS URI '{imas_uri}': no IDS name given. Hint: "
            "add '#<idsname>' to your URI."
        )
    if occurrence:
        try:
            occurrence = int(occurrence)
        except ValueError:
            raise click.UsageError(
                f"Invalid IMAS URI '{imas_uri}': "
                f"occurrence '{occurrence}' is not an integer."
            ) from None
    else:
        occurrence = 0

    logging.info("Opening data entry...")
    with imas.DBEntry(base_uri, "r") as entry:
        logging.info("Reading IDS...")
        # Ensure IDS uses homogeneous time, extract all time points
        lazy_ids = entry.get(idsname, occurrence, lazy=True, autoconvert=False)
        if lazy_ids.ids_properties.homogeneous_time != IDS_TIME_MODE_HOMOGENEOUS:
            raise click.ClickException("The loaded IDS is not using homogeneous time.")
        times = lazy_ids.time[:]
        del lazy_ids
        logging.info("Found %d time slices to stream", len(times))
        if n and n < len(times):
            logging.info("Streaming first %d time slices", n)
            times = times[:n]
        n = len(times)

        # Get first time slice to obtain the static and metadata
        ids = entry.get_slice(
            idsname, times[0], CLOSEST_INTERP, occurrence, autoconvert=False
        )
        ids_producer = StreamingIDSProducer(ids)
        kafka_producer = KafkaProducer(
            KafkaSettings(host=kafka_host, topic_name=kafka_topic),
            ids_producer.metadata,
        )

        if get:
            logging.info("Loading full IDS...")
            ids = entry.get(idsname, occurrence)
            logging.info("IDS loaded.")

            with click.progressbar(
                ids_producer.messages_from_batch(ids),
                length=n,
                label="Streaming time slices",
                show_pos=True,
                update_min_steps=_PROGRESS_BAR_UPDATE_MINSTEP,
            ) as bar:
                for i, data in enumerate(bar):
                    if i == n:
                        break
                    kafka_producer.produce(bytes(data))
                # Make bar go to 100%
                bar.make_step(n % _PROGRESS_BAR_UPDATE_MINSTEP)
                bar.render_progress()
            return

        # Send remaining time slices
        with click.progressbar(
            times, label="Streaming time slices", show_pos=True
        ) as bar:
            for time in bar:
                ids = entry.get_slice(
                    idsname,
                    time,
                    CLOSEST_INTERP,
                    occurrence,
                    autoconvert=False,
                    lazy=True,
                )
                kafka_producer.produce(bytes(ids_producer.create_message(ids)))


@main.command()
@click.argument("kafka_host")
@click.argument("kafka_topic")
@click.argument("imas_uri")
@click.option(
    "--batch-size", default=100, help="Number of time slice to batch per put_slice."
)
@click.option(
    "--overwrite", is_flag=True, help="Overwrite any existing IMAS Data Entry."
)
@click.option("--timeout", "-t", default=5.0, help="Timeout for receiving next message")
def kafka_to_imasentry(
    kafka_host: str,
    kafka_topic: str,
    imas_uri: str,
    batch_size: int,
    overwrite: bool,
    timeout: float,
):
    """Consume streaming IMAS data from Kafka and store data in an IMAS Data Entry.

    \b
    Arguments:
        KAFKA_HOST  Kafka host and port (aka bootstrap.servers). E.g. 'localhost:9092'.
        KAFKA_TOPIC Name of the kafka topic with streaming IMAS data.
        IMAS_URI    IMAS URI to store the data at, for example 'imas:hdf5?path=./out'.
                    The program will not overwrite existing data (unless the --overwrite
                    flag is given). Only backends that implement 'put_slice' are
                    supported, such as HDF5 and MDSPLUS.
    """
    # Local import: kafka is an optional dependency
    from imas_streams.kafka import KafkaConsumer, KafkaSettings

    consumer = KafkaConsumer(
        KafkaSettings(host=kafka_host, topic_name=kafka_topic),
        BatchedIDSConsumer,
        batch_size=batch_size,
        return_copy=False,
    )

    mode = "w" if overwrite else "x"
    with imas.DBEntry(imas_uri, mode) as entry:
        for result in consumer.stream(timeout=timeout):
            if result is not None:
                entry.put_slice(result)


@main.command()
def kafka_to_muscle3():
    """MUSCLE3 actor consuming streaming IMAS data from a Kafka topic and making it
    available to a MUSCLE3 workflow.
    """
    # Ensure optional dependencies are available
    try:
        import libmuscle  # noqa: F401

        import imas_streams.kafka  # noqa: F401
    except ModuleNotFoundError:
        click.echo("Error: please install the optional kafka and muscle3 dependencies.")
        sys.exit(1)

    from imas_streams.muscle3_datasource import data_source

    data_source()
