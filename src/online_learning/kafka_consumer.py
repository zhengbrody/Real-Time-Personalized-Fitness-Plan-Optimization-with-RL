"""
Kafka Consumer for Online Learning

Consumes user feedback events from Kafka and processes them
through the OnlineLearningLoop for continuous model updates.
"""

import json
import logging
import os
import signal
import sys
from typing import Optional

from src.online_learning.loop import OnlineLearningLoop

logger = logging.getLogger(__name__)

# Graceful shutdown flag
_shutdown = False


def _handle_signal(signum, frame):
    """Handle termination signals for graceful shutdown."""
    global _shutdown
    logger.info("Received signal %s, shutting down gracefully...", signum)
    _shutdown = True


class FeedbackConsumer:
    """
    Kafka consumer that reads user feedback messages and feeds them
    into the OnlineLearningLoop for online model updates.
    """

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        topic: str = "training.user.feedback",
        group_id: str = "online-learning-consumer",
    ):
        """
        Initialize the Kafka feedback consumer.

        Args:
            bootstrap_servers: Kafka broker addresses (comma-separated).
                               Defaults to KAFKA_BOOTSTRAP_SERVERS env var
                               or "localhost:9092".
            topic: Kafka topic to consume from.
            group_id: Consumer group id.
        """
        self.bootstrap_servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self.topic = topic
        self.group_id = group_id
        self.consumer = None
        self.loop = OnlineLearningLoop()

    def _connect(self):
        """Create the KafkaConsumer, with a graceful fallback if unavailable."""
        try:
            from kafka import KafkaConsumer

            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                consumer_timeout_ms=5000,
            )
            logger.info(
                "Connected to Kafka at %s, consuming topic '%s'",
                self.bootstrap_servers,
                self.topic,
            )
        except ImportError:
            logger.error(
                "kafka-python package is not installed. "
                "Install it with: pip install kafka-python"
            )
            raise
        except Exception as e:
            logger.error("Failed to connect to Kafka at %s: %s", self.bootstrap_servers, e)
            raise

    def _process_message(self, message) -> None:
        """
        Process a single Kafka message through the OnlineLearningLoop.

        Expected message value schema::

            {
                "user_id": "user_001",
                "action_id": 3,
                "feedback": {
                    "completion": 1,
                    "rpe": 7,
                    "mood": 4,
                    ...
                }
            }
        """
        try:
            data = message.value
            user_id = data.get("user_id")
            action_id = data.get("action_id")
            feedback = data.get("feedback")

            if user_id is None or action_id is None or feedback is None:
                logger.warning(
                    "Skipping malformed message (missing required fields): %s", data
                )
                return

            reward = self.loop.process_feedback(
                user_id=user_id,
                action_id=int(action_id),
                feedback=feedback,
            )
            logger.info(
                "Processed feedback for user=%s action=%s reward=%.4f",
                user_id,
                action_id,
                reward,
            )
        except Exception as e:
            logger.error("Error processing message: %s", e, exc_info=True)

    def run(self) -> None:
        """
        Start consuming messages in a loop until a shutdown signal is received
        or an unrecoverable error occurs.
        """
        global _shutdown

        self._connect()

        logger.info("Consumer loop started. Waiting for messages...")
        try:
            while not _shutdown:
                # poll() honours consumer_timeout_ms; returns an iterator
                for message in self.consumer:
                    if _shutdown:
                        break
                    self._process_message(message)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            if self.consumer is not None:
                self.consumer.close()
                logger.info("Kafka consumer closed")

    def close(self) -> None:
        """Close the consumer connection."""
        if self.consumer is not None:
            self.consumer.close()
            self.consumer = None
            logger.info("Kafka consumer closed")


def main():
    """Entry point for the Kafka consumer (used by Dockerfile CMD)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info("Starting Kafka feedback consumer...")

    try:
        consumer = FeedbackConsumer()
        consumer.run()
    except Exception as e:
        logger.error("Consumer failed to start: %s", e)
        sys.exit(1)

    logger.info("Consumer shut down cleanly")


if __name__ == "__main__":
    main()
