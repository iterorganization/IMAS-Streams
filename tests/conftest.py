import os

import pytest


@pytest.fixture
def kafka_host():
    value = os.getenv("KAFKA_HOST")
    if not value:
        pytest.fail("Cannot connect to Kafka server: KAFKA_HOST not set.")
    return value
