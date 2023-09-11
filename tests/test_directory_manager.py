import os
import sys
sys.path.append("..")

import unittest
from apps.ml.directory_manager import DirectoryManager

class TestDirectoryManager(unittest.TestCase):

    def test_find_base_dir(self):
        # Assuming a directory named "loyalwingmen" exists in the current path
        path = os.path.abspath(os.path.dirname(__file__))
        self.assertTrue(os.path.exists(DirectoryManager.find_base_dir(path)))

    def test_get_base_dir(self):
        self.assertTrue(os.path.exists(DirectoryManager.get_base_dir()))

    def test_get_outputs_dir(self):
        app_name = "test_app"
        self.assertTrue(os.path.exists(DirectoryManager.get_outputs_dir(app_name)))

    def test_get_logs_dir(self):
        outputs_dir = DirectoryManager.get_outputs_dir("test_app")
        self.assertIsNotNone(DirectoryManager.get_logs_dir(outputs_dir))

    def test_get_models_dir(self):
        outputs_dir = DirectoryManager.get_outputs_dir("test_app")
        self.assertIsNotNone(DirectoryManager.get_models_dir(outputs_dir))

    def test_create_output_folder(self):
        experiment_name = "test_experiment"
        outputs_dir = DirectoryManager.get_outputs_dir("test_app")
        folder_name = DirectoryManager.get_outputs_dir(experiment_name)
        self.assertTrue(os.path.exists(folder_name))

if __name__ == "__main__":
    unittest.main()
