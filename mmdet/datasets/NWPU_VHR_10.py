from .coco import CocoDataset

class NWPU_VHR_10(CocoDataset):

    CLASSES = ('airplane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court',
           'basketball-court', 'ground-track-field', 'harbor', 'bridge', 'vehicle',)