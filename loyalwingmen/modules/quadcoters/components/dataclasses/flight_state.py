from dataclasses import dataclass, field
import numpy as np
from typing import Union, List, Dict, Any


class FlightStateManager:
    def __init__(self):
        # Initialize a dictionary to store sensor readings
        self.reset()

    def reset(self):
        """Reset the flight state to default values."""
        self.data = {}
        
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
            return {key: self.data.get(key)}  # This will return the value for the key if it exists, or None if it doesn't
        
        # If key is a list of strings, return a dictionary of corresponding values for existing keys.
        if isinstance(key, list):
            return {k: self.data.get(k) for k in key}
        
        return {}
        
    def get_inertial_data(self) -> Dict[str, Any]:
        """Retrieve data related to the inertial measurement unit (IMU)."""
        inertial_keys = ["position", "velocity", "attitude", "quaternions", "angular_rate"]
        return self.get_data(inertial_keys) or {key: None for key in inertial_keys}


    # Additional states can be added as needed, such as:
    # - Accelerations (linear and angular)
    # - Control surface deflections (aileron, elevator, rudder, etc.)
    # - Engine settings or thrust levels
    # - etc.
