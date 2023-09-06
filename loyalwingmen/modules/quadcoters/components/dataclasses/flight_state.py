from dataclasses import dataclass, field
import numpy as np
from typing import Union, List, Dict, Any


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

    def get_data(self, key: Union[str, List[str], None] = None) -> Dict[str, Any]:
        """Retrieve specific sensor readings or all data."""
        
        # If key is None or an empty string, return a copy of the entire data.
        if not key:
            return self.data.copy()
        
        # If key is a string, check if it exists and then return the corresponding value.
        if isinstance(key, str):
            return {key: self.data[key]} if key in self.data else {}
        
        # If key is a list of strings, return a dictionary of corresponding values for existing keys.
        if isinstance(key, list):
            return {k: self.data[k] for k in key if k in self.data}

    # Additional states can be added as needed, such as:
    # - Accelerations (linear and angular)
    # - Control surface deflections (aileron, elevator, rudder, etc.)
    # - Engine settings or thrust levels
    # - etc.
