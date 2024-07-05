import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.transforms import BaseTransform  # type: ignore


class AddZeroNodeAttr(BaseTransform):
    r"""To make it easy to run GNNs that expect node_attr on graphs without them, this graph transformation gives every node a zero node feature / attribute.

    Args:
        edge_attr_size (int): Length of the attributes that will be added to each edge
    """

    def __init__(self, node_attr_size: int = True):
        assert node_attr_size > 0
        self.node_attr_size = node_attr_size

    def __call__(self, data: Data) -> Data:
        data.x = torch.zeros((data.num_nodes, self.node_attr_size), dtype=torch.int32)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(node_attr_size={self.node_attr_size})"
