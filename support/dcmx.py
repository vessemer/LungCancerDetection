import dicom
import json
from skimage.draw import polygon
from os import listdir, walk
import tensorflow as tf
from os.path import join, isfile
from scipy.misc import imresize

from numpy import *

class DICOMReader:
    def extract_dicom(self, files, verbose=1):
        images = []
        # loop through all the DICOM files
        for i, filenameDCM in enumerate(files):
            # read the file
            if verbose:
                print("step: " + filenameDCM + " ", i)
            ds = dicom.read_file(filenameDCM)
            # store the raw image data
            images += [ds.pixel_array]
        return images
    

    def collect_dcm_filenames(self, path):
        train_files = []
        for root, folders, files in walk(path):
            for file in files:
                if file.split('.')[-1].lower() in ['dcm', 'dicom']:
                    train_files.append(join(root, file))
        return train_files

    
    def dcm_to_tfr(self, files, path='lungs.tfrecords', scale=2, verbose=1):
        writer = tf.python_io.TFRecordWriter(path)
        for file in files:
            img = self.extract_dicom([file], verbose)[0]
            img = imresize(img, size=(array(img.shape) / scale).astype(int), interp='bicubic').astype(uint8)
            height = img.shape[0]
            width = int(img.shape[1] / 2)

            img = [img[:, :width], fliplr(img[:, width:])]

            for lung in img:
                lung_raw = lung.tostring()

                example = tf.train.Example(features=tf.train.Features(feature={
                            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[lung.shape[0]])),
                            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[lung.shape[1]])),
                            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lung_raw]))}))

                writer.write(example.SerializeToString())

        writer.close()


    def left_right(self, data, side):
        lung = [data[side + 'LungContour']['XCoordinates'], \
                         data[side + 'LungContour']['YCoordinates']]

        lung[0] += [lung[0][0]]
        lung[1] += [lung[1][0]]

        return lung


    def combine(self, data, shape):
        mask = zeros((shape[0] + 1, shape[1] + 1))
        left = self.left_right(data ,'Left')
        right = self.left_right(data ,'Right')
        try:
            x, y = polygon(right[1], right[0])
            mask[x, y] = 1
            x, y = polygon(left[1], left[0])
            mask[x, y] = 1
        except:
            print('Throwed')
            pass
        return mask


    def extract_polygons(self, files, images):
        masks = []
        for img, polygon, i in zip(images, files, range(len(files))):
            print("step: " + polygon + " ", i)
            with open(polygon) as json_data:
                data = json.load(json_data)
            masks += [self.combine(data, shape(img))]
        return masks


    def extract_data(self, start = 0, end = 1, PathDicom = './', extract_lungs=True):
        lstFilesDCM = []  # create an empty list
        lstFilesJSON = []  # create an empty list
        print(PathDicom)
        print(walk(PathDicom))
        for dirName, subdirList, fileList in walk(PathDicom):
            for filename in fileList:
                if ".dcm" in filename.lower():
                    lstFilesDCM.append(join(dirName,filename))
                if ".json" in filename.lower(): 
                    lstFilesJSON.append(join(dirName,filename))

        end = min(len(lstFilesDCM), end)
        if end == -1:
            end = len(lstFilesDCM)

        images = self.extract_dicom(sorted(lstFilesDCM)[start:end])
        
        if extract_lungs:
            masks = self.extract_polygons(sorted(lstFilesJSON)[start:end], images)
        else:
            masks = []

        return images, masks
    
    
    def cut_on_patches(self, images, masks, patch_size=28):
        patches = []
        for img, mask in zip(images, masks):
            for x in range(0, shape(mask)[0] - patch_size, int(patch_size / 4.)):
                for y in range(0, shape(mask)[1] - patch_size, int(patch_size / 4.)):
                    if sum(mask[x:x+patch_size, y:y+patch_size]) > patch_size:
                        tmp = shape(img[x:x+patch_size, y:y+patch_size])
                        if tmp[0] == patch_size and tmp[1] == patch_size: 
                            patches += [img[x:x+patch_size, y:y+patch_size].reshape(patch_size, patch_size, 1)]
                        else:
                            print(tmp)
        return patches
