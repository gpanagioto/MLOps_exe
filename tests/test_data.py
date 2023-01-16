from src.data.make_dataset import CorruptMnist
from tests import _PATH_DATA, _PROJECT_ROOT, _TEST_ROOT
import pytest
import os

def test_data():
   test_dataset= _PATH_DATA + "/processed/test_pt"
   train_dataset = _PATH_DATA + "/processed/train_pt"
   assert len(train_dataset) == 25000 or len(train_dataset) == 40000, "Dataset did not have the correct number of samples" 
   assert len(test_dataset) == 5000

