from src.data.make_dataset import CorruptMnist
import os
from tests import _PATH_DATA
import pytest

in_folder = _PATH_DATA + "/raw"


N_train = 40000  # corrupted v2 : 40000, otherwise 25000
N_test = 5000
img_tensor_dims = [1, 28, 28]


@pytest.mark.skipif(not os.path.exists(in_folder), reason="Data files not found")
def test_train_dataset_dims():
    train_dataset = CorruptMnist(train=True, in_folder=in_folder, out_folder="")
    assert train_dataset.data.shape[1] == img_tensor_dims[0]
    assert train_dataset.data.shape[2] == img_tensor_dims[1]
    assert train_dataset.data.shape[3] == img_tensor_dims[2]
    assert train_dataset.data.shape[0] == N_train


@pytest.mark.skipif(not os.path.exists(in_folder), reason="Data files not found")
def test_test_dataset_dims():
    test_dataset = CorruptMnist(train=False, in_folder=in_folder, out_folder="")
    assert test_dataset.data.shape[0] == N_test
    assert test_dataset.data.shape[1] == img_tensor_dims[0]
    assert test_dataset.data.shape[2] == img_tensor_dims[1]
    assert test_dataset.data.shape[3] == img_tensor_dims[2]


@pytest.mark.skipif(not os.path.exists(in_folder), reason="Data files not found")
def test_labels_represented():
    # I know it is mnist, so the number of distinct labels should be 10
    train_dataset = CorruptMnist(train=True, in_folder=in_folder, out_folder="")
    test_dataset = CorruptMnist(train=False, in_folder=in_folder, out_folder="")
    assert len(train_dataset.targets.unique()) == 10
    assert len(test_dataset.targets.unique()) == 10
    assert (train_dataset.targets.unique() == test_dataset.targets.unique()).all()

