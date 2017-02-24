""""
Lung segmentation was taken from
https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
"""
from numpy import *
from skimage import measure, morphology, segmentation
from skimage.segmentation import clear_border
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import ball, disk, dilation, binary_erosion, \
    remove_small_objects, erosion, closing, reconstruction, binary_closing
import scipy.ndimage as ndimage
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi


def largest_label_volume(im, bg=-1):
    vals, counts = unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True, sobel=False):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = array(image > -320, dtype=int8) + 1
    labels = measure.label(binary_image)

    # Creation of the Sobel-Gradient
    if sobel:
        sobel_gradient = gradient(image)
        sobel_gradient = linalg.norm(sobel_gradient, axis=0)
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 2

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0

    if sobel:
        return binary_image.astype(bool_), \
               (sobel_gradient * binary_image).astype(float16)

    return binary_image.astype(bool_)


"""
Lung segmentation was taken from
https://www.kaggle.com/arnavkj95/data-science-bowl-2017/candidate-generation-and-luna16-preprocessing/discussion
"""


def get_lungs_mask(im):
    binary = im < 604
    cleared = clear_border(binary)
    label_image = label(cleared)
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    selem = disk(10)
    binary = binary_closing(binary, selem)
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    get_high_vals = binary == 0
    im[get_high_vals] = 0
    return im != 0


def segment_lung_from_ct_scan(ct_scan):
    return asarray([get_lungs_mask(slice) for slice in ct_scan])



"""
Lung segmentation was taken from
https://www.kaggle.com/ankasor/data-science-bowl-2017/improved-lung-segmentation-using-watershed/discussion
"""


def generate_markers(image):
    # Creation of the internal Marker
    marker_internal = image < -400
    img_shape = image.shape
    marker_internal = segmentation.clear_border(marker_internal)
    marker_internal_labels = measure.label(marker_internal)
    areas = [r.area for r in measure.regionprops(marker_internal_labels)]
    areas.sort()
    if len(areas) > 2:
        for region in measure.regionprops(marker_internal_labels):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    marker_internal_labels[coordinates[0], coordinates[1]] = 0
    marker_internal = marker_internal_labels > 0
    # Creation of the external Marker
    external_a = ndimage.binary_dilation(marker_internal, iterations=10)
    external_b = ndimage.binary_dilation(marker_internal, iterations=55)
    marker_external = external_b ^ external_a
    # Creation of the Watershed Marker matrix
    marker_watershed = zeros(img_shape, dtype=int)
    marker_watershed += marker_internal * 255
    marker_watershed += marker_external * 128

    return marker_internal, marker_external, marker_watershed


def exclude_lungs(image):
    # Creation of the markers as shown above:
    marker_internal, marker_external, marker_watershed = generate_markers(image)
    img_shape = image.shape

    # Creation of the Sobel-Gradient
    sobel_filtered_dx = ndimage.sobel(image, 1)
    sobel_filtered_dy = ndimage.sobel(image, 0)
    sobel_gradient = hypot(sobel_filtered_dx, sobel_filtered_dy)
    sobel_gradient *= 255.0 / sobel_gradient.max()

    # Watershed algorithm
    watershed = morphology.watershed(sobel_gradient, marker_watershed)

    # Reducing the image created by the Watershed algorithm to its outline
    outline = ndimage.morphological_gradient(watershed, size=(3, 3))
    outline = outline.astype(bool)

    # Performing Black-Tophat Morphology for reinclusion
    # Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)
    # Perform the Black-Hat
    outline += ndimage.black_tophat(outline, structure=blackhat_struct)

    # Use the internal marker and the Outline that was just created to generate the lungfilter
    lungfilter = bitwise_or(marker_internal, outline)
    # Close holes in the lungfilter
    # fill_holes is not used here, since in some slices the heart would be reincluded by accident
    lungfilter = ndimage.morphology.binary_closing(lungfilter, structure=ones((5, 5)), iterations=3)

    # Apply the lungfilter (note the filtered areas being assigned -2000 HU)
    segmented = where(lungfilter == 1, image, -2000 * ones(img_shape))

    return segmented, lungfilter, sobel_gradient