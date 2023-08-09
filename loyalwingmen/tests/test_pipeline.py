import os
import sys
sys.path.append("..")

import unittest
from apps.ml.pipeline import ReinforcementLearningPipeline

class TestExtractFolderInfo(unittest.TestCase):
    def test_valid_folder_name(self):
        folder_name = "PPO-h[892, 712, 284, 887]-f15-lr4.017315358240058e-09-r4624539.5-sd23697994.0"
        expected = {
            'model_name': 'PPO',
            'hidden_layers': [892, 712, 284, 887],
            'rl_frequency': 15,
            'learning_rate': 4.017315358240058e-09,
            'avg_reward': 4624539.5,
            'reward_std_dev': 23697994.0
        }
        actual = ReinforcementLearningPipeline.extract_folder_info(folder_name)
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
