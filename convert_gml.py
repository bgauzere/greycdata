"""
Conversion for molecules graphs between `torch_geometric.data.Data` and `gml` file format
"""

import os
import sys
import zipfile
import networkx as nx
import torch
from torch_geometric.data import Data

def data_to_gml(data: Data, output: str) -> None:
    """
    Writes a `Data` object into a `gml` file

    Parameters :
    ------------

    * `data`   : `Data` object to save
    * `output` : output file path
    """
    g = nx.Graph()

    for index, features in enumerate(data.x):
        g.add_node(index, x=features.tolist())

    for index, (src, dst) in enumerate(zip(data.edge_index[0], data.edge_index[1])):
        g.add_edge(
            src.item(),
            dst.item(),
            edge_attr=data.edge_attr[index].item()
        )

    g.graph["y"] = data.y.item()
    nx.write_gml(g, output)

def gml_to_data(gml: str) -> Data:
    """
    Reads a `gml` file and creates a `Data` object

    Parameters :
    ------------

    * `gml` : gml file path

    Returns :
    ---------

    * `Data` : the `Data` object created from the file content
    """
    g: nx.Graph = nx.read_gml(gml)
    x, edge_index, edge_attr = [], [], []

    y = torch.tensor([g.graph["y"]], dtype=torch.long) if "y" in g.graph else None

    for _, attr in g.nodes(data=True):
        x.append(attr["x"])

    for u, v, attr in g.edges(data=True):
        edge_index.append([int(u), int(v)])
        edge_index.append([int(v), int(u)])
        edge_attr.append(attr["edge_attr"])

    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    return Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
