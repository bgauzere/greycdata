
import torch
from torch_geometric.data import InMemoryDataset, download_url
from greycdata.converter import load_alkane_pyg_graphs, load_acyclic_pyg_graphs


# https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html#creating-in-memory-datasets
class GreycDataset(InMemoryDataset):
    _dataset_paths = {'Alkane': 'graphkit-learn/datasets/Alkane/Alkane/',
                      'Acyclic': 'graphkit-learn/datasets/Acyclic/'}

    def __init__(self, root, name: str, transform=None, pre_transform=None, pre_filter=None):
        #super().__init__(root, transform, pre_transform, pre_filter)
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []  # ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def _load_data(self):
        if self.name == 'Alkane':
            return load_alkane_pyg_graphs()
        elif self.name == 'Acyclic':
            return load_acyclic_pyg_graphs()
        else:
            raise Exception("Dataset not found")

    def process(self):
        # Read data into huge `Data` list.
        data_list = self._load_data()
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
