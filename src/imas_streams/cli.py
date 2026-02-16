import logging

import click
import imas

from imas_streams import BatchedIDSConsumer


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
@click.argument("kafka_host")
@click.argument("kafka_topic")
@click.argument("imas_uri")
@click.option(
    "--batch-size", default=100, help="Number of time slice to batch per put_slice."
)
@click.option(
    "--overwrite", is_flag=True, help="Overwrite any existing IMAS Data Entry."
)
def kafka_to_imasentry(
    kafka_host: str, kafka_topic: str, imas_uri: str, batch_size: int, overwrite: bool
):
    """Consume streaming IMAS data from Kafka and store data in an IMAS Data Entry.

    N.B. This program requires the optional kafka dependency.

    \b
    Arguments:
        KAFKA_HOST  Kafka host and port (aka bootstrap.servers). E.g. 'localhost:9092'
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
        for result in consumer.stream():
            if result is not None:
                entry.put_slice(result)
