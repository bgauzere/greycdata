# Installation of required packages

1. Install pytorch geometric >= 2.0
see https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

3. `pipenv install`

# Contents
## greycdata : 
    |- datasets.py : Implementation of three GREYC chemistry small datasets as pytorch
    geometric datasets : Alkane, Acyclic and MAO. See https://brunl01.users.greyc.fr/CHEMISTRY/ for details
     - loaders.py : load the same datasets as list of networkx graphs
     
# Examples

Two notebooks, one for classification, one for regression, are provided for testing
purposes

# Authors
- Benoit Gaüzère <benoit.gauzere@insa-rouen.fr>
- Linlin Jia <https://github.com/jajupmochi>
