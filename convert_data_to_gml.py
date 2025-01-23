"""
Conversion for molecules graphs between `torch_geometric.data.Data` and `gml` file format
"""

import os
from typing import Optional, List
import shutil
import zipfile
from tqdm import tqdm
import networkx as nx
import torch
from torch_geometric.data import Data
from greycdata.datasets import GreycDataset
from greycdata.metadata import GREYC_META

GML_SEPARATOR = "---"

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
            edge_attr=data.edge_attr[index].tolist()
        )

    g.graph["y"] = data.y.item()

    if output is None:
        content = ""
        gml = nx.generate_gml(g)
        for line in gml:
            content += line + '\n'
        return content

    nx.write_gml(g, output)
    return None

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
        g: nx.Graph = nx.parse_gml(gml)

    x, edge_index, edge_attr = [], [], []

    y = torch.tensor([g.graph["y"]], dtype=torch.long) if "y" in g.graph else None

    for _, attr in g.nodes(data=True):
        x.append(attr["x"])

    for u, v, attr in g.edges(data=True):
        edge_index.append([int(u), int(v)])
        edge_index.append([int(v), int(u)])
        edge_attr.append(attr["edge_attr"])
        edge_attr.append(attr["edge_attr"])

    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    return Data(x=x, edge_attr=edge_attr, edge_index=edge_index, y=y)

def dataset_to_gml(dataset_name: str, output: str) -> None:
    """
    Coverts a whole dataset from `GreycDataset` into a gml format

    Parameters :
    ------------

    * `dataset_name` : name of the greyc dataset
    * `output`       : name of the output gml file (will be zipped if it ends with `.zip`)
    """
    dataset_dir = f"{dataset_name}_tmp"
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    dataset = GreycDataset(dataset_dir, dataset_name)
    gml_contents = [data_to_gml(data) for data in tqdm(dataset)]

    fname = f"{os.path.splitext(output)[0]}.gml"
    with open(fname, "w", encoding="utf8") as f:
        for gml_content in gml_contents[:-1]:
            f.write(gml_content)
            f.write(f"\n{GML_SEPARATOR}\n")
        f.write(gml_contents[-1])

    if os.path.splitext(output)[1] == ".zip":
        with zipfile.ZipFile(f"{os.path.splitext(output)[0]}.zip", 'w') as zipf:
            zipf.write(fname, fname.lower())
        os.remove(fname)
    else:
        os.rename(fname, output)

    print(f"Dataset {dataset_name} fully converted")
    shutil.rmtree(dataset_dir)

def gml_to_dataset(gml: str) -> List[Data]:
    """
    Reads a dataset from a gml file and converts it into a list of `Data`

    Parameters :
    ------------

    * `gml` : Source filename. If `zip` file, it will be extracted

    Returns :
    ---------

    * `List[Data]` : Content of the gml dataset
    """
    if os.path.splitext(gml)[1] == ".zip":
        with zipfile.ZipFile(gml, 'r') as zipf:
            gml_file = zipf.namelist()[0]
            zipf.extract(gml_file, os.getcwd())
        gml = gml_file

    with open(gml, 'r', encoding="utf8") as f:
        gml_contents = f.read()

    gml_files = gml_contents.split(GML_SEPARATOR)

    return [gml_to_data(content, False) for content in gml_files]

def main() -> None:
    """Main script"""
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    dst = os.path.join(current_dir, "greycdata/data_gml")
    if not os.path.exists(dst):
        os.mkdir(dst)
    datasets = GREYC_META.keys()
    for dataset in datasets:
        print(f"======= Converting {dataset} =======")
        dataset_to_gml(dataset, os.path.join(dst, f"{dataset.lower()}.zip"))

if __name__ == "__main__":
    main()
