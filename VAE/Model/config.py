PATH = {
    'DATA': '/home/vess/Projects/ML/Lung/data/whole/TbDispensary',
    'TRAIN': '/home/vess/Projects/ML/Lung/data/whole/TbDispensary/lungs_train.tfrecords',
    'VALID': '/home/vess/Projects/ML/Lung/data/whole/TbDispensary/lungs_valid.tfrecords'
}


LUNG_SHAPE = {
    'HEIGHT': 75,
    'WIDTH': 40
}


DATA_TRANSFORMATION = {
    'SCALE': 3.
}


TRAIN_PARAMS = {
    'NUM_EPOCHS': None,
    'SHUFFLE': True,
    'VALIDATION_PART': .2
}

