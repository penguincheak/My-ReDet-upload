from .coco import CocoDataset

class GaoFen_challengeDataset(CocoDataset):

    CLASSES = ('other', 'A220', 'Boeing737', 'A321', 'Boeing787', 'Boeing747',
               'A330', 'Boeing777', 'A350', 'ARJ21')