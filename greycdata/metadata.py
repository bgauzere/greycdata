"""
Module defining metadata for the GREYC datasets.

This metadata dictionary provides information for various datasets, including:
- Task type (e.g., regression or classification)
- File names for datasets and targets
- Data and graph formats
- Separator used for target values
- Extra parameters specific to certain datasets
"""

GREYC_META = {
    'Acyclic': {
        'task_type': 'regression',
        'filename_dataset': 'dataset_bps.ds',
        'filename_targets': None,
        'dformat': 'ds',
        'gformat': 'ct',
        'y_separator': ' ',
    },
    'Alkane': {
        'task_type': 'regression',
        'filename_dataset': 'dataset.ds',
        'filename_targets': 'dataset_boiling_point_names.txt',
        'dformat': 'ds',
        'gformat': 'ct',
        'y_separator': ' ',
    },
    'MAO': {
        'task_type': 'classification',
        'filename_dataset': 'dataset.ds',
        'filename_targets': None,
        'dformat': 'ds',
        'gformat': 'ct',
        'y_separator': ' ',
    },
    'Monoterpens': {
        'task_type': 'classification',
        'filename_dataset': 'dataset_ct.ds',
        'filename_targets': None,
        'dformat': 'ds',
        'gformat': 'ct',
        'y_separator': ' ',
    },
    'PAH': {
        'task_type': 'classification',
        'filename_dataset': 'dataset.ds',
        'filename_targets': None,
        'dformat': 'ds',
        'gformat': 'ct',
        'y_separator': ' ',
    },
}