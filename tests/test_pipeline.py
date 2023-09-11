import os
import sys
sys.path.append("..")

import unittest
from apps.ml.pipeline import ReinforcementLearningPipeline
import pandas as pd

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
        
        
    def test_save_results_to_excel(self):
        # Save the test data to a temporary Excel file
        temp_file = 'temp_test.xlsx'
        
        # Clean up: remove the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file) 
            
        # Define test data
        test_results = {'hidden_1': 256, 'hidden_2': 256, 'hidden_3': 256, 'rl_frequency': 5, 'learning_rate': 1e-08, 'speed_amplification': 10, 'model': 'ppo', 'avg_score': -434563.25, 'std_deviation': 275.81005859375}
        
        
        
        ReinforcementLearningPipeline.save_results_to_excel('.', temp_file, test_results)
        
        # Load the saved Excel file
        saved_df = pd.read_excel(temp_file)
        
        # Define the expected DataFrame
        expected_df = pd.DataFrame([test_results])
        
        # Perform assertions to compare saved and expected DataFrames
        print(expected_df)
        print(saved_df)
        self.assertTrue(saved_df.equals(expected_df))
        
        # Clean up: remove the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)   


if __name__ == "__main__":
    unittest.main()
