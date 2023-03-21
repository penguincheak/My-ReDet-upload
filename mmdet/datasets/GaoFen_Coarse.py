from .coco import CocoDataset

class GaoFenCoarseDataset(CocoDataset):

    CLASSES = ('Vehicle', 'Ship', 'Airplane')