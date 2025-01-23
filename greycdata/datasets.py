import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx
from greycdata.loaders import load_dataset
from greycdata.metadata import GREYC_META


class GreycDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for loading GREYC molecular datasets.

    This class allows loading five GREYC datasets ('Acyclic', 'Alkane', 'MAO', 
    'Monoterpens', 'PAH') as PyTorch Geometric datasets, converting molecular 
    graphs from NetworkX  format to PyTorch Geometric `Data` objects.

    See `"GREYC's Chemistry dataset" <https://lucbrun.ensicaen.fr/CHEMISTRY/>`_
    for details.

    Parameters
    ----------
    root : str
        Root directory where the dataset will be stored.
    name : str
        Name of the GREYC dataset to load ('Acyclic', 'Alkane', 'MAO', 
        'Monoterpens' or 'PAH').
    transform : callable, optional
        A function/transform to apply to each data object during runtime.
    pre_transform : callable, optional
        A function/transform to apply to each data object before saving.
    pre_filter : callable, optional
        A function to filter data objects. If provided, only the data objects 
        for which this function returns True will be saved.

    Examples
    --------
    >>> dataset = GreycDataset(root='/tmp/Acyclic/', name='Acyclic')
    >>> len(dataset)
    183  # Example dataset size
    >>> dataset[0]  # Access the first PyTorch Geometric Data object
    Data(edge_index=[2, 4], bond_type=[4], bond_stereo=[4], x=[3, 7], y=[1])
    """

    def __init__(
            self, 
            root, 
            name: str, 
            transform=None, 
            pre_transform=None, 
            pre_filter=None
        ):
        if name not in GREYC_META:
            raise ValueError(f"Dataset '{name}' Not Found.")
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        # Deletion of edge attributes duplicates
        self.data.edge_attr = torch.unique(self.data.edge_attr, dim=1)

    def __repr__(self):
        return f"{self.name}({len(self)})"

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def process(self):
        # Read data into huge `Data` list.
        graph_list, property_list, self.atom_list = load_dataset(self.name, 
                                                                 True)

        # Convert to PyG.
        def from_nx_to_pyg(graph, y):
            """
            Converts Networkx Graph to a PyTorch Graph and adds y
            """
            node_attrs, edge_attrs = None, None
            for node in graph.nodes:
                node_attrs = graph.nodes[node].keys()
                break
            if graph.edges:
                for edge in graph.edges:
                    edge_attrs = graph.edges[edge].keys()
                    break
            pyg_graph = from_networkx(
                graph, 
                group_node_attrs=node_attrs,
                group_edge_attrs=edge_attrs,
            )
            pyg_graph.y = y
            return pyg_graph

        data_list = [from_nx_to_pyg(graph, y)
                     for graph, y in zip(graph_list, property_list)]
        
        # Add edge_attr of size 0 to all graphs that don't have it.
        # (Only if the dataset has edge attributes)
        no_edge_attr_data = []
        dataset_has_edge_attr = False
        for data in data_list:
            if data.edge_attr is None:
                if dataset_has_edge_attr:
                    data.edge_attr = torch.tensor([])
                else:
                    no_edge_attr_data.append(data)
            else:
                dataset_has_edge_attr = True
        if dataset_has_edge_attr:
            for data in no_edge_attr_data:
                data.edge_attr = torch.tensor([])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
