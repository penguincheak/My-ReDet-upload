from .coco import CocoDataset

class GaoFenDataset(CocoDataset):

    CLASSES = ('Passenger Ship', 'Motorboat', 'Fishing Boat', 'Tugboat',
                'Engineering Ship', 'Liquid Cargo Ship',
               'Dry Cargo Ship', 'Warship', 'Small Car', 'Van', 'Dump Truck',
               'Cargo Truck', 'Intersection', 'Truck Tractor',
                'Bus', 'Tennis Court', 'Trailer', 'Excavator',
               'A220', 'Football Field', 'Boeing737', 'Baseball Field', 'A321',
               'Boeing787', 'Basketball Court', 'Boeing747', 'A330', 'Boeing777',
               'Tractor', 'Bridge', 'A350', 'C919', 'ARJ21', 'Roundabout')
    # 'other-ship', 'other-vehicle', 'other-airplane',