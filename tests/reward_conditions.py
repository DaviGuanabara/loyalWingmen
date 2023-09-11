import sys
import os
sys.path.append("..")

import enum
import time
from modules.models.lidar import Channels
from math import pi
import random
import unittest

class YourClass:
    @staticmethod
    def _getValueForChannel(features, channel):
        values = [value for ch, _, _, value in features if ch == channel]
        return values[0] if values else None

    @staticmethod
    def _isTimeout(reset_time, *args, **kwargs):
        current = time.time()
        return current - reset_time > 20

    @staticmethod
    def _isWithinRadius(reset_time, features, radius, *args, **kwargs):
        value = YourClass._getValueForChannel(features, Channels.DISTANCE_CHANNEL.value)
        if value is not None:
            distance = value * radius
            return distance < radius
        return False

    @staticmethod
    def checkConditions(reset_time, features, radius, condition_handlers):
        triggered_conditions = []

        for condition_name, check_function in condition_handlers.items():
            if check_function(reset_time, features, radius):
                triggered_conditions.append(condition_name)

        return triggered_conditions





def generate_features(num_pairs=5):
    features = []
    for _ in range(num_pairs):
        theta = random.uniform(0, pi)
        phi = random.uniform(-pi, pi)

        # Generate values for channels 0 and 1
        value_ch0 = random.uniform(0, 1)
        value_ch1 = random.choice([0, 1, 2, 3])

        # Channel 0 feature
        features.append((0, theta, phi, value_ch0))

        # Channel 1 feature
        features.append((1, theta, phi, value_ch1))

    return features



class TestYourClass(unittest.TestCase):

    def setUp(self):
        self.radius = 5
        self.your_class_instance = YourClass()
        self.condition_handlers = {
            "timeout": YourClass._isTimeout,
            "within_radius": YourClass._isWithinRadius
        }

    def test_timeout_condition(self):
        reset_time, features = self.your_setup_for_timeout_condition()
        conditions = self.your_class_instance.checkConditions(reset_time, features, self.radius, self.condition_handlers)
        self.assertIn("timeout", conditions, "Timeout condition should be triggered")

    def test_within_radius_condition(self):
        reset_time, features = self.your_setup_for_within_radius_condition()
        conditions = self.your_class_instance.checkConditions(reset_time, features, self.radius, self.condition_handlers)
        self.assertIn("within_radius", conditions, "Within_radius condition should be triggered")

    def test_no_conditions(self):
        reset_time, features = self.your_setup_for_no_conditions()
        conditions = self.your_class_instance.checkConditions(reset_time, features, self.radius, self.condition_handlers)
        self.assertFalse(conditions, "No conditions should be triggered")

    def your_setup_for_timeout_condition(self):
        reset_time = time.time() - 25 # Assuming 20 seconds is the timeout threshold
        features = generate_features()
        return reset_time, features

    def your_setup_for_within_radius_condition(self):
        reset_time = time.time() - 10 # Within the timeout threshold
        features = generate_features()
        # Modify the distance value for one of the channels to be within the radius
        features[0] = (features[0][0], features[0][1], features[0][2], 0.1) # Assuming radius is 5
        return reset_time, features

    def your_setup_for_no_conditions(self):
        reset_time = time.time() - 10 # Within the timeout threshold
        features = generate_features()
        # Ensure the distance value for all the channels is outside the radius
        features = [(ch, theta, phi, 1.0) for ch, theta, phi, _ in features]
        return reset_time, features

if __name__ == '__main__':
    unittest.main()
