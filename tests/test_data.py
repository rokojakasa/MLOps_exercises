import os.path

import pytest
import torch
from mlops.data import corrupt_mnist

from tests import _PATH_DATA


# @pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
# def test_my_dataset():
#     """Test the MyDataset class."""
#     train_set, test_set = corrupt_mnist()
#     assert len(train_set) == 50000
#     assert len(test_set) == 5000
#     for dataset in [train_set, test_set]:
#         for x, y in dataset:
#             assert x.shape == (1, 28, 28)
#             assert y in range(10)

#     train_targets = torch.unique(train_set.tensors[1])
#     assert (train_targets == torch.arange(0, 10)).all()
#     test_targets = torch.unique(test_set.tensors[1])
#     assert (test_targets == torch.arange(0, 10)).all()
