import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.utils import from_networkx
from greycdata.loaders import load_alkane, load_acyclic, load_MAO

# https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-in-memory-datasets


class DatasetNotFoundError(Exception):
    pass


class GreycDataset(InMemoryDataset):
    """
    Class to load three GREYC Datasets as pytorch geometric dataset
    """

    def __init__(self, root, name: str, transform=None, pre_transform=None, pre_filter=None):
        """
        name : Acyclic, Alkane or MAO, depending on dataset to load
        """
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # @property
    # def raw_file_names(self):
    #     return []  # ['some_file_1', 'some_file_2', ...]

    def __str__(self):
        return self.name

    @property
    def processed_file_names(self):
        return ['data.pt']

    def _load_data(self):
        """
        Load the right data according to initializer
        """
        if self.name == 'Alkane':
            return load_alkane()
        elif self.name == 'Acyclic':
            return load_acyclic()
        elif self.name == 'MAO':
            return load_MAO()
        else:
            raise DatasetNotFoundError("Dataset not found")

    def process(self):
        # Read data into huge `Data` list.
        graph_list, property_list = self._load_data()
        # Convert to PyG.

        def from_nx_to_pyg(graph, y):
            """
            Convert networkx graph to pytorch graph and add y
            """
            pyg_graph = from_networkx(graph, group_node_attrs=[
                                      'atom_symbol', 'degree', 'x', 'y', 'z'])
            pyg_graph.y = y
            return pyg_graph

        data_list = [from_nx_to_pyg(graph, y)
                     for graph, y in zip(graph_list, property_list)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
