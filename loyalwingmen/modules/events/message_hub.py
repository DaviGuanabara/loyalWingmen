from .broker import Broker
from typing import Dict, Callable, Any

class MessageHub:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MessageHub, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize attributes for the instance."""
        self._brokers: Dict[str, Broker] = {}

    def get_broker(self, topic: str) -> Broker:
        """Get the broker for a specific topic. Creates a new one if it doesn't exist."""
        if topic not in self._brokers:
            self._brokers[topic] = Broker()
        return self._brokers[topic]

    def subscribe(self, topic: str, subscriber: Callable[[Any, int], None]) -> None:  # <-- Updated type hint
        broker = self.get_broker(topic)
        broker.subscribe(topic, subscriber)

    def unsubscribe(self, topic: str, subscriber: Callable[[Any, int], None]) -> None:  # <-- Updated type hint
        broker = self.get_broker(topic)
        broker.unsubscribe(topic, subscriber)

    def publish(self, topic: str, message: Any, publisher_id: int) -> None:
        broker = self.get_broker(topic)
        broker.publish(topic, message, publisher_id)
