# Installation of Required Packages

1. Install PyTorch Geometric (version >= 2.0).  
   Refer to the official [installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

2. Install dependencies using Pipenv:  
   ```bash
   pipenv install
   ```

# Contents

## `greycdata`  
This folder contains tools to work with GREYC chemistry datasets.  

- **`datasets.py`**  
  Implementation of five [GREYC chemistry datasets](https://lucbrun.ensicaen.fr/CHEMISTRY/) as PyTorch Geometric datasets:  
  - Alkane  
  - Acyclic  
  - MAO  
  - Monoterpenes  
  - PAH  

- **`loaders.py`**  
  Provides functionality to load the same datasets as lists of NetworkX graphs.

# Examples

Two example notebooks are provided for testing purposes:
- **Classification**  
- **Regression**

# Authors

- Benoit Gaüzère - [benoit.gauzere@insa-rouen.fr](mailto:benoit.gauzere@insa-rouen.fr)  
- Linlin Jia - [GitHub Profile](https://github.com/jajupmochi)