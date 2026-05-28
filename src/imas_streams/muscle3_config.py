import sys

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

    imas_streams:
        executable: {sys.executable}
        args: -m imas_streams dynamic-kafka-to-muscle3
        description: |
            # Data source for multiple IMAS-Streams data streams

            This is a data source that reads Streaming IMAS data from multiple Kafka
            topics and makes the data available in a MUSCLE3 simulation.

            ## Usage

            To use this component in your simulation, you need to configure the
            following:

            1. Define output ports for each IMAS Stream for the O_I operator.
            2. Specify the Kafka host to connect to in the `kafka_host` setting.
            3. Specify the Kafka topic names for each output port in the `kafka_topics`
               setting as `<output_port>: <kafka_topic>`.
            
            ## Example configuration

            ```yaml
            imports:
            - from imas_streams.data_source import implementation imas_streams
            models:
              example:
                streams:
                  description: IMAS Streams data source and sink
                  implementation: imas_streams
                  ports:
                    O_I: magnetics_out pf_active_out
                    S: equilibrium_in
                equilibrium:
                  description: Equilibrium reconstruction code
                  ports:
                    F_INIT: magnetics_in pf_active_in
                    O_F: equilibrium_out
                conduits:
                  streams.magnetics_out: equilibrium.magnetics_in
                  streams.pf_active_out: equilibrium.pf_active_in
                  equilibrium.equilibrium_out: streams.equilibrium_in
            settings:
              streams.kafka_host: localhost:9092
              streams.kafka_topics: |
                magnetics_out: kafka.topic.for.magnetics
                pf_active_out: kafka.topic.for.pf_active
                equilibrium_in: kafka.topic.for.equilibrium
            ```
            
            In this example, IMAS Streams from two topics are made available:

            1. The data on the topic `kafka.topic.for.magnetics` is sent on the output
               port `magnetics_out`.
            2. The data on the topic `kafka.topic.for.pf_active` is sent on the output
               port `pf_active_out`.

            The data received on the S port `equilibrium_in` is published to
            `kafka.topic.for.equilibrium`.
        supported_settings:
            kafka_host: >
                str Bootstrap server address for Kafka (e.g. "localhost:9092" for a
                locally running kafka).
            kafka_topics: >
                str List of kafka topics per output / input port, in the form of
                `port_name: topic_name`. Each entry must be on a separate line.
            kafka_timeout: float Timeout when receiving Kafka messages.
"""
"""yMMSL description of the imas_streams MUSCLE3 components"""
