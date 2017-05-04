import sys, os

sys.path.append('../')
from numpy import *


from glob import glob
from tqdm import tqdm
from os.path import join, basename
from skimage.measure import label
from sklearn.cluster import KMeans
from path import *


def crop_and_not_rotate(segmenteds):
    x, y, z = where(segmenteds)
    if len(x):
        segmenteds = segmenteds[x.min():x.max(), y.min():y.max(), z.min():z.max()]
    else:
        segmenteds = array([])
    return segmenteds


def segment_nodules(patches, masks, is_nodule=True, magic_const=50):
    missed = list()
    nodules_cropped = list()

    for patch_path, mask_path in tqdm(zip(patches, masks)):
        patch = load(patch_path)
        mask = load(mask_path)

        if not min(patch.shape):
            missed.append(patch_path)
            continue

        prepared = patch
        prepared = (prepared - prepared.min()) / (prepared.max() - prepared.min())

        kmeans = KMeans(n_clusters=2)
        if IS_NODULE:
            coords = where(mask == 2)
        else:
            coords = where(mask >= 0)
        data = prepared[coords]
        data = kmeans.fit_predict(expand_dims(data, 1))
        kmean = zeros(mask.shape)
        kmean[coords] = data + magic_const
        labels, num = label(kmean, return_num=True, background=0)

        nodule_a = argmax([sum(labels == i) for i in range(1, num + 1)]) + 1
        init = kmeans.predict(expand_dims(prepared[labels == nodule_a], 1)).min()
        nodule_b = list()
        for i in range(1, num + 1):
            if i != nodule_a:
                if kmeans.predict(expand_dims(prepared[where(labels == i)], 1)).min() != init:
                    nodule_b.append((sum(labels == i), i))

        nodule_b = max(nodule_b)[1]

        A = prepared[labels == nodule_a]
        B = prepared[labels == nodule_b]

        if mean(A.reshape(-1)) > mean(B.reshape(-1)):
            labels = labels == nodule_a
        else:
            labels = labels == nodule_b

        mask[mask == 1] = 0
        mask += labels  # 0 - hyunya, 1 - vessel, 2 - nodule, 3 - intersection
        cropped = crop_and_not_rotate(labels)
        nodules_cropped.append(cropped.shape)
        save(join(NODULES,
                  basename(patch_path).split('patch.npy')[0] + 'mask_nodules'), labels)


NODULES = PATH['LUNA_NODULES']
IS_NODULE = True
patches = sort(glob(join(NODULES, '*patch.npy')))
masks = sort(glob(join(NODULES, '*mask.npy')))
segment_nodules(patches, masks, IS_NODULE)
