"""
Module to load greyc datasets as list of networkx graphs
"""

import os
from enum import Enum
from greycdata.file_managers import DataLoader
from greycdata.utils import one_hot_encode
from greycdata.metadata import GREYC_META


PATH = os.path.dirname(__file__)


def get_atom_list(graphs):
    atom_set = set()
    for graph in graphs:
        for node in graph.nodes:
            atom_set.add(graph.nodes[node]['atom_symbol'])
    return list(atom_set)


def prepare_graph(
        graph, 
        atom_list=['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H']
    ):
    """
    Prepare graph to include all data before pyg conversion
    Parameters
    ----------
    graph : networkx graph
        graph to be prepared
    atom_list : list[str], optional
        List of node attributes to be converted into one hot encoding
    """
    # Convert attributes.
    graph.graph = {}
    for node in graph.nodes:
        graph.nodes[node]['atom_symbol'] = one_hot_encode(
            graph.nodes[node]['atom_symbol'],
            atom_list,
            include_unknown_set=True,
        )
        graph.nodes[node]['degree'] = float(graph.degree[node])
        for attr in ['x', 'y', 'z']:
            graph.nodes[node][attr] = float(graph.nodes[node][attr])

    for edge in graph.edges:
        for attr in ['bond_type', 'bond_stereo']:
            graph.edges[edge][attr] = float(graph.edges[edge][attr])

    return graph


def load_dataset(dataset_name: str):
    """Load the dataset as a llist of networkx graphs and returns list of 
    graphs and list of properties.

    Args:
       dataset_name:str the dataset to load (Alkane,  Acyclic, ...)

    Returns:
       List of nx graphs
       List of properties (float or int)
    """
    if dataset_name not in GREYC_META:
        raise Exception("Dataset Not Found")
    
    loader = loader_dataset(dataset_name)

    atom_list = get_atom_list(loader.graphs)
    graphs = [prepare_graph(graph, atom_list) for graph in loader.graphs]

    return graphs, loader.targets


def loader_dataset(dataset_name: str):
    """
    Load the n graphs of `dataset_name` datasets.
    Returns two lists:
    - The n networkx graphs
    - Boiling points
    """
    metadata = GREYC_META[dataset_name]
    
    ds_path = os.path.join(PATH, "data", dataset_name)
    if metadata["filename_targets"] is not None:
        filename_targets = os.path.join(ds_path, metadata["filename_targets"])
    else:
        filename_targets = None

    dloader = DataLoader(
        os.path.join(ds_path, metadata["filename_dataset"]),
        filename_targets=filename_targets,
        dformat=metadata["dformat"], 
        gformat=metadata["gformat"], 
        y_separator=metadata["y_separator"])
    
    if metadata["task_type"] == "classification":
        dloader._targets = [int(yi) for yi in dloader.targets]

    return dloader