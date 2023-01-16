from src.models.model import MyAwesomeModel
import torch
import numpy as np
import pytest

model = MyAwesomeModel()


def test_forward():
    # Test for batch size 1
    batch_size = 1
    rand_input = torch.rand([batch_size, 1, 28, 28])
    assert (
        model(rand_input).detach().numpy().shape == np.array([batch_size, 10])
    ).all(), "Sample fed to model does not have the correct dimensions"
    # Test for batch size other than 1, eg 64
    batch_size = 64
    rand_input = torch.rand([batch_size, 1, 28, 28])
    assert (
        model(rand_input).detach().numpy().shape == np.array([batch_size, 10])
    ).all(), "Data sample fed to model does not have the corect dimensions"


def test_on_wrong_shape():
    with pytest.raises(ValueError, match="Expected a tensor of size 4"):
        model(torch.randn(1, 2, 3))
    with pytest.raises(
        ValueError, match="Expected each sample to have dimensions \[1,28,28\]"
    ):
        model(torch.randn(1, 1, 27, 27))

