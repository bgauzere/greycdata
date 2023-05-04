"""
Module to load greyc datasets as list of networkx graphs
"""

import os
from enum import Enum
from greycdata.file_managers import DataLoader
from greycdata.utils import one_hot_encode


class DatasetName(str, Enum):
    ALKANE = "Alkane"
    ACYCLIC = "Acyclic"
    MAO = "MAO"


PATH = os.path.dirname(__file__)


def prepare_graph(graph, atom_list=['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H']):
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


def _load_greyc_networkx_graphs(dataset: DatasetName):
    """Load the dataset as a llist of networkx graphs and returns list of graphs and list of properties

    Args:
       dataset:str the dataset to load (Alkane,Acyclic,...)

    Returns:
       list of nx graphs
       list of properties (float or int)
    """
    loaders = {DatasetName.ALKANE: _loader_alkane,
               DatasetName.ACYCLIC: _loader_acyclic,
               DatasetName.MAO: _loader_mao}
    loader_f = loaders.get(dataset, None)
    loader = loader_f()
    if loader is None:
        raise Exception("Dataset Not Found")

    graphs = [prepare_graph(graph) for graph in loader.graphs]
    return graphs, loader.targets


def load_alkane():
    return _load_greyc_networkx_graphs(DatasetName.ALKANE)


def load_acyclic():
    return _load_greyc_networkx_graphs(DatasetName.ACYCLIC)


def load_MAO():
    return _load_greyc_networkx_graphs(DatasetName.MAO)


def _loader_alkane():
    """
    Load the 150 graphs of Alkane datasets
    returns two lists
    - 150 networkx graphs
    - boiling points
    """
    # Load dataset.
    rel_path = 'data/Alkane/'
    ds_path = os.path.join(PATH, rel_path)
    dloader = DataLoader(
        os.path.join(ds_path, 'dataset.ds'),
        filename_targets=os.path.join(
            ds_path, 'dataset_boiling_point_names.txt'),
        dformat='ds', gformat='ct', y_separator=' ')
    return dloader


def _loader_acyclic():
    # Load dataset.
    rel_path = 'data/Acyclic/'
    ds_path = os.path.join(PATH, rel_path)
    dloader = DataLoader(
        os.path.join(ds_path, 'dataset_bps.ds'),
        filename_targets=None,
        dformat='ds', gformat='ct', y_separator=' ')
    return dloader


def _loader_mao():
    # Load dataset.
    rel_path = 'data/MAO/'
    ds_path = os.path.join(PATH, rel_path)
    dloader = DataLoader(
        os.path.join(ds_path, 'dataset.ds'),
        filename_targets=None,
        dformat='ds', gformat='ct', y_separator=' ')
    dloader._targets = [int(yi) for yi in dloader.targets]
    return dloader
