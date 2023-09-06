from dataclasses import dataclass, field
import numpy as np


class FlightState:
    def __init__(self):
        # Initialize a dictionary to store sensor readings
        self.data = {
            # "position": np.zeros(3),
            # "velocity": np.zeros(3),
            # "attitude": np.zeros(3),
            # "quaternions": np.zeros(4),
            # "angular_rate": np.zeros(3),
            # ... other predefined fields ...
        }

    def update_data(self, sensor_data: dict):
        """Update the flight state with new sensor data."""
        self.data.update(sensor_data)

    def get_data(self, key: str = ""):
        """Retrieve a specific sensor reading or all data."""
        return self.data.get(key) if key else self.data.copy()

    # Additional states can be added as needed, such as:
    # - Accelerations (linear and angular)
    # - Control surface deflections (aileron, elevator, rudder, etc.)
    # - Engine settings or thrust levels
    # - etc.
