"""
Conversion for molecules graphs between `torch_geometric.data.Data` and `gml` file format
"""

import os
import sys
import io
from typing import Optional
import zipfile
import networkx as nx
import torch
from torch_geometric.data import Data

def data_to_gml(data: Data, output: Optional[str] = None) -> Optional[str]:
    """
    Writes a `Data` object into a `gml` file

    Parameters :
    ------------

    * `data`   : `Data` object to save
    * `output` : output file path (Default value `None`).
        If `None` is given, will return the file content instead

    Returns :
    ---------

    * `str`  : The gml file content if no output file was given
    * `None` : If an output file was given
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

    if output is None:
        output = io.StringIO()
    nx.write_gml(g, output)

    if isinstance(output, io.StringIO):
        result = output.getvalue()
        output.close()
        return result

def gml_to_data(gml: str, gml_file: bool = True) -> Data:
    """
    Reads a `gml` file and creates a `Data` object

    Parameters :
    ------------

    * `gml`      : gml file path if `gml_file` is set to `True`, gml content otherwise
    * `gml_file` : indicates whether `gml` is a path to a gml file or a gml file content

    Returns :
    ---------

    * `Data` : the `Data` object created from the file content

    Raises :
    --------

    * `FileNotFoundError` : if `gml` is a file path and doesn't exist
    """
    if gml_file:
        if not os.path.exists(gml):
            raise FileNotFoundError(f"File `{gml}` does not exist")
        g: nx.Graph = nx.read_gml(gml)
    else:
        buffer = io.StringIO(gml)
        g: nx.Graph = nx.read_gml(buffer)
        buffer.close()

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
