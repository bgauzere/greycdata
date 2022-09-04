import os
from greycdata.utils import one_hot_encode
from torch_geometric.utils import from_networkx
from gklearn.dataset import DataLoader


def convert_graph(graph,
                  atom_list=['C', 'N', 'O', 'F',
                             'P', 'S', 'Cl', 'Br', 'I', 'H'],
                  g_property=None):
    '''
    Convert a molecular nx graph to a pytorch geometric graph
    '''
    # Convert attributes.
    graph.graph = {}
    for node in graph.nodes:
        graph.nodes[node]['atom_symbol'] = one_hot_encode(
            graph.nodes[node]['atom_symbol'],
            atom_list,
            include_unknown_set=True,
        )
        graph.nodes[node]['degree'] = float(graph.degree[node])
        # for attr in ['x', 'y', 'z']:
        #     del graph.nodes[node][attr]
        for attr in ['x', 'y', 'z']:
            graph.nodes[node][attr] = float(graph.nodes[node][attr])
    for edge in graph.edges:
        for attr in ['bond_type', 'bond_stereo']:
            graph.edges[edge][attr] = float(graph.edges[edge][attr])

    # Convert to PyG.
    pyg_graph = from_networkx(graph, group_node_attrs=[
                              'degree', 'x', 'y', 'z'])
    pyg_graph.y = g_property
    return pyg_graph


def load_alkane_pyg_graphs():
    # Load dataset.
    ds_path = 'graphkit-learn/datasets/Alkane/Alkane/'
    dloader = DataLoader(
        os.path.join(ds_path, 'dataset.ds'),
        filename_targets=os.path.join(
            ds_path, 'dataset_boiling_point_names.txt'),
        dformat='ds', gformat='ct', y_separator=' ')

    graphs = dloader.graphs  # networkx graphs
    return [convert_graph(graph, g_property=y) for graph, y in zip(graphs, dloader._targets)]
