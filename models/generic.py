from typing import List

import torch
from torch import Tensor
from torch.nn import functional as F


def pad_stack_tensors(matrices: List[Tensor], pad_value=0) -> Tensor:
    if len(matrices) == 0:
        return None

    max_dims = [max(mat.size(dim) for mat in matrices) for dim in range(matrices[0].dim())]
    result = []
    for mat in matrices:
        padding_shape = []
        for dim in range(mat.dim()):
            padding_shape.insert(0, max_dims[dim] - mat.size(dim))
            padding_shape.insert(0, 0)
        result.append(F.pad(mat, padding_shape, value=pad_value))
    return torch.stack(result)
