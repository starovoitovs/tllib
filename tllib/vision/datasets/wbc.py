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
    }

    CLASSES = ['basophil', 'eosinophil', 'erythroblast', 'myeloblast', 'promyelocyte', 'myelocyte', 'metamyelocyte',
               'neutrophil_banded', 'neutrophil_segmented', 'monocyte', 'lymphocyte_typical']

    crop_sizes = {
        'A': 250,
        'M': 345,
        'W': 288,
    }

    def __init__(self, root: str, task: str, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])
        super(WBC, self).__init__(root, WBC.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())
