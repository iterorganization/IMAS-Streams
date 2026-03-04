import logging
import sys

from imas_streams import BatchedIDSConsumer

logger = logging.getLogger(__name__)


DATA_SOURCE = f"""
ymmsl_version: v0.2

description: Importable yMMSL configuration for imas_streams_source
programs:
    imas_streams_source:
        executable: {sys.executable}
        args: -m imas_streams kafka-to-muscle3

        ports:
            o_i: ids_out
            s: trigger

        description: |
            # IMAS-Streams data source

            Data source reading Streaming IMAS data from a Kafka topic and making it
            available in a MUSCLE3 simulation.

            The `ids_out` port sends one message for every `batch_size` time slices
            streamed over the configured kafka topic. The type of IDS depends on the
            configured kafka topic: please take care that this matches the IDS that is
            expected for components receiving the message.

            You may use the `trigger` port to indicate that the previous message is
            processed and a new message may be sent. If this port is not connected then
            this component will send messages on the `ids_out` port as soon as they are
            available.

        supported_settings:
            kafka_host: >
                str Bootstrap server address for Kafka (e.g. "localhost:9092" for a
                locally running kafka).
            kafka_topic: >
                str Name of the kafka topic with streaming IMAS data to subscribe to.
            batch_size: >
                int Number of time slices to batch in a single MUSCLE3 message.
                Default is one time slice per message.
            most_recent_only: >
                bool If not set, or set to false, all data in the IMAS Data Stream is
                provided to the MUSCLE3 simulation.
                This can be set to true to provide the last available time point with
                each iteration. This mode is useful while data is being produced (e.g.
                during an experimental pulse) and it is more important to have
                up-to-date data than to process all time points.
"""
"""yMMSL description of the imas_streams_source actor"""


def data_source():
    # Local imports for all optional dependencies
    from libmuscle import Instance, Message
    from ymmsl import Operator

    from imas_streams.kafka import KafkaConsumer, KafkaSettings

    logger.info("Creating libmuscle instance")
    instance = Instance({Operator.O_I: ["ids_out"], Operator.S: ["trigger"]})

    while instance.reuse_instance():
        logger.info("Reading settings")
        kafka_host = instance.get_setting("kafka_host", "str")
        kafka_topic = instance.get_setting("kafka_topic", "str")
        batch_size = instance.get_setting("batch_size", "int", default=1)
        most_recent_only = instance.get_setting(
            "most_recent_only", "bool", default=False
        )
        if most_recent_only and batch_size != 1:
            raise RuntimeError(
                "'Most recent only' mode is incompatible with a batch size larger "
                "than 1. Please update the MUSCLE3 settings to set 'most_recent_only: "
                "false', or 'batch_size: 1'."
            )
        if most_recent_only and not instance.is_connected("trigger"):
            raise RuntimeError(
                "'Most recent only' mode requires that the 'trigger' port is connected."
            )

        logger.info("Creating kafka consumer")
        consumer = KafkaConsumer(
            KafkaSettings(host=kafka_host, topic_name=kafka_topic),
            BatchedIDSConsumer,
            # FIXME: depends on https://github.com/iterorganization/IMAS-Streams/pull/11
            # most_recent_only=most_recent_only,
            batch_size=batch_size,
        )

        logger.info("Start reading from the IMAS data stream")
        for ids in consumer.stream():
            if ids is None:
                continue  # Batch is not yet complete

            serialized = ids.serialize()
            time = ids.time[0]
            msg = Message(time, data=serialized)
            instance.send("ids_out", msg)

            # Optional: wait for trigger message
            if instance.is_connected("triggger"):
                instance.receive("trigger")
        logger.info("IMAS data stream ended")

    logger.info("Reuse loop finished")
