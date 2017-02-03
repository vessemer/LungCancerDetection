from scipy.ndimage import filters
from numpy import *

class EnergyNormalization:
    def decompose(self, images, B=7, alpha=2):
        decomposed = []
        blured = [images]
        for i in range(B):
            blured += [[filters.gaussian_filter(img, alpha ** (i)) for img in images]]
            decomposed += [[old  - new
                           for old, new in zip(blured[i], blured[i + 1])]]
        decomposed += [blured[-1]]
        return list(zip(*decomposed))


    def normalize(self, decomposed, masks, immutable=[], immutable_masks=[]):
        energies = [[std(layer[mask > 0]) 
                     for layer in img] 
                    for img, mask in zip(decomposed, masks)]

        immutable_energies = [[std(layer[mask > 0]) 
                     for layer in img] 
                    for img, mask in zip(immutable, immutable_masks)]

        referenced = asarray([e.mean() 
                              for e in asarray(immutable_energies).T])

        images = []
        diff = []
        for imgs, energy in zip(decomposed, energies):
            normalized = zeros(imgs[0].shape)
            for layer, e, ref in zip(imgs, energy, referenced):
                diff += [ref / e]
                normalized += layer * diff[-1]
            images += [normalized]

        return images, diff


    def iterate_normalization(self, images, masks, immutable=[], immutable_masks=[], n_iterations=10, verbose=True):
        decomposed_immutable = self.decompose(immutable)
        for i in range(n_iterations):
            decomposed = self.decompose(images)
            images, diff = self.normalize(decomposed, masks, decomposed_immutable, immutable_masks) 
            if verbose:
                print('Step: ', i, ', diff: ', mean(diff))
        return images
