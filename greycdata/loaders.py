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
    """
    Extract a list of unique atom symbols from a list of NetworkX graphs.

    Parameters
    ----------
    graphs : list[networkx.Graph]
        A list of NetworkX graphs, each containing nodes with an 'atom_symbol' 
        attribute.

    Returns
    -------
    list[str]
        A list of unique atom symbols found in the nodes of the graphs.

    Examples
    --------
    >>> import networkx as nx
    >>> g1 = nx.Graph()
    >>> g1.add_node(0, atom_symbol='C')
    >>> g1.add_node(1, atom_symbol='H')
    >>> g2 = nx.Graph()
    >>> g2.add_node(0, atom_symbol='O')
    >>> g2.add_node(1, atom_symbol='C')
    >>> get_atom_list([g1, g2])
    ['C', 'H', 'O']
    """
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
    Prepares a NetworkX graph by encoding node and edge attributes.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to be prepared. Nodes must have 'atom_symbol', 'x', 'y', 
        and 'z' attributes. Edges must have 'bond_type' and 'bond_stereo' 
        attributes.
    atom_list : list[str], optional
        A list of atom symbols to encode using one-hot encoding. Default is 
        ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'H'].

    Returns
    -------
    networkx.Graph
        The modified graph with encoded attributes.

    Examples
    --------
    >>> import networkx as nx
    >>> g = nx.Graph()
    >>> g.add_node(0, atom_symbol='C', x=0.0, y=0.0, z=0.0)
    >>> g.add_node(1, atom_symbol='O', x=1.0, y=1.0, z=1.0)
    >>> g.add_edge(0, 1, bond_type=1, bond_stereo=0)
    >>> prepare_graph(g, ['C', 'O', 'H'])
    <networkx.classes.graph.Graph at ...>
    >>> g.nodes[0]['atom_symbol']
    [1.0, 0.0, 0.0] # One-hot encoding of 'C'
    >>> g.nodes[1]['atom_symbol']
    [0.0, 1.0, 0.0] # One-hot encoding of 'O'
    >>> g.edges[(0, 1)]['bond_type']
    1.0
    """
    # Convert attributes.

    graph.graph = {}
    for node in graph.nodes:
        graph.nodes[node]['atom_symbol'] = one_hot_encode(
            graph.nodes[node]['atom_symbol'],
            atom_list,
        ) if len(atom_list) > 1 else []
        graph.nodes[node]['degree'] = float(graph.degree[node])
        for attr in graph.nodes[node]:
            if attr not in ['atom_symbol', 'degree']:
                graph.nodes[node][attr] = float(graph.nodes[node][attr])

    for edge in graph.edges:
        for attr in graph.edges[edge]:
            graph.edges[edge][attr] = float(graph.edges[edge][attr])

    return graph


def load_dataset(dataset_name: str, return_atom_list=False):
    """
    Loads and prepares (encodes node and edge attributes) a dataset of 
    molecular graphs and their associated properties.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to load (e.g., 'Alkane', 'Acyclic', etc.).
    return_atom_list : bool, optional
        If True, the function also returns the list of unique atom symbols 
        found in the dataset. Default is False.

    Returns
    -------
    tuple
        If `return_atom_list` is False:
            (list[networkx.Graph], list[float or int])
            - A list of processed NetworkX graphs.
            - A list of target properties (e.g., boiling points).
        If `return_atom_list` is True:
            (list[networkx.Graph], list[float or int], list[str])
            - The above two elements, plus a list of unique atom symbols.

    Raises
    ------
    ValueError
        If the dataset name is not found in the metadata.

    Examples
    --------
    >>> graphs, targets = load_dataset("Alkane")
    >>> len(graphs)
    150  # Example output
    >>> targets[0]
    -164.0  # Example boiling point

    >>> _, _, atom_list = load_dataset("Acyclic", return_atom_list=True)
    >>> atom_list
    ['C', 'S', 'O']  # Example atom symbols
    """
    if dataset_name not in GREYC_META:
        raise ValueError(f"Dataset {dataset_name} Not Found.")
    
    loader = loader_dataset(dataset_name)

    atom_list = get_atom_list(loader.graphs)
    graphs = [prepare_graph(graph, atom_list) for graph in loader.graphs]

    if return_atom_list:
        return graphs, loader.targets, atom_list
    return graphs, loader.targets


def loader_dataset(dataset_name: str):
    """
    Loads a dataset of molecular graphs and their corresponding properties.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to load (e.g., 'Alkane', 'Acyclic', etc.).

    Returns
    -------
    DataLoader
        An object containing:
        - A list of NetworkX graphs representing the molecules.
        - A list of target properties (e.g., boiling points or classes).

    Examples
    --------
    >>> loader = loader_dataset("Alkane")
    >>> len(loader.graphs)
    150  # Example number of graphs
    >>> loader.targets[0]
    -164.0  # Example boiling point
    """
    metadata = GREYC_META[dataset_name]
    
    ds_path = os.path.join(PATH, "data", dataset_name)
    if metadata["filename_targets"] is not None:
        filename_targets = os.path.join(ds_path, metadata["filename_targets"])
    else:
        filename_targets = None

    if 'extra_params' in metadata:
        kwargs = metadata['extra_params']
    else:
        kwargs = {}

    dloader = DataLoader(
        os.path.join(ds_path, metadata["filename_dataset"]),
        filename_targets=filename_targets,
        dformat=metadata["dformat"], 
        gformat=metadata["gformat"], 
        y_separator=metadata["y_separator"],
        **kwargs
    )
    
    if metadata["task_type"] == "classification":
        dloader._targets = [int(yi) for yi in dloader.targets]

    return dloader