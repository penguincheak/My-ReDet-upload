from .coco import CocoDataset

class DIORDataset(CocoDataset):

    CLASSES = ('airplane', 'airport', 'baseballfield',
               'basketballcourt', 'bridge', 'chimney',
               'dam', 'Expressway-Service-area',
               'Expressway-toll-station', 'harbor',
               'golffield', 'groundtrackfield', 'overpass',
               'ship', 'stadium', 'storagetank',
               'tenniscourt', 'trainstation', 'vehicle',
               'windmill')