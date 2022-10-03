"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Optional
import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class WBC(ImageList):

    image_list = {
        "A": "image_list/ace.txt",
        "M": "image_list/mat.txt",
        "W": "image_list/wbc.txt",
        "T": "image_list/wbc2.txt",
    }

    CLASSES = ['basophil', 'eosinophil', 'erythroblast', 'myeloblast', 'promyelocyte', 'myelocyte', 'metamyelocyte',
               'neutrophil_banded', 'neutrophil_segmented', 'monocyte', 'lymphocyte_typical']

    crop_sizes = {
        'A': 300,
        'M': 345,
        'W': 288,
        'T': 288,
    }

    norm_means = {
        'A': [0.8691, 0.7415, 0.7161],
        'M': [0.8209, 0.7282, 0.8364],
        'W': [0.7404, 0.6518, 0.7791],
        'T': [0.7404, 0.6518, 0.7791],
    }

    norm_stds = {
        'A': [0.1635, 0.1909, 0.0789],
        'M': [0.1649, 0.2523, 0.0945],
        'W': [0.1884, 0.2513, 0.1654],
        'T': [0.1884, 0.2513, 0.1654],
    }

    def __init__(self, root: str, task: str, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])
        super(WBC, self).__init__(root, WBC.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())
