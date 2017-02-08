from segmentation import *
from support.ct_reader import *
from paths import *
from glob import glob
from os.path import join
from tqdm import tqdm


patients = glob(join(PATH['DATA'], '*'))

for patient in tqdm(patients):
    ct_scan, file_names = read_ct_scan(patient)

    ct_scan, slice_heights = extract_array(ct_scan)
    ct_mask = segment_lung_from_ct_scan(ct_scan)

    save(join(patient, 'mask'), ct_mask)

