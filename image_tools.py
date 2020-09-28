
from cellpose import models as cellpose_models

from skimage.segmentation import find_boundaries
from skimage.measure import label, regionprops
import numpy as np

def get_coordinates(tile, plot=False):
    if not tile.flags.writeable:
        tile = tile.copy()
    log_tile = np.log(tile)
    log_tile[np.isinf(log_tile)] = 10 ** -16
    # DEFINE CELLPOSE MODEL
    # model_type='cyto' or model_type='nuclei'
    cellpose = cellpose_models.Cellpose(gpu=False, model_type='nuclei', net_avg=False)

    # define CHANNELS to run segementation on
    # grayscale=0, R=1, G=2, B=3
    # channels = [cytoplasm, nucleus]
    # if NUCLEUS channel does not exist, set the second channel to 0
    # channels = [0,0]
    # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
    # channels = [0,0] # IF YOU HAVE GRAYSCALE
    # channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
    # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

    # or if you have different types of channels in each image
    channels = [[0, 0]]

    # if diameter is set to None, the size of the cells is estimated on a per image basis
    # you can set the average cell `diameter` in pixels yourself (recommended)
    # diameter can be a list or a single number for all images
    masks, flows, styles, diams = cellpose.eval([log_tile], diameter=20, channels=channels)

    if plot == True:
        import matplotlib.pyplot as plt
        outlines = np.zeros(masks[0].shape, np.bool)
        outlines[find_boundaries(masks[0], mode='inner')] = 1
        outX, outY = np.nonzero(outlines)
        imgout = tile.copy()
        imgout[outX, outY] = np.array([np.max(tile)])
        plt.imshow(imgout)

    label_img = label(masks[0])
    regions = regionprops(label_img)
    list_x_coords = []
    list_y_coords = []
    for example_region in regions:
        width = 101
        height = 101
        rmin, cmin, rmax, cmax = example_region.bbox
        x_position = int(example_region.centroid[1])
        y_position = int(example_region.centroid[0])
        list_x_coords.append(x_position)
        list_y_coords.append(y_position)
    return [list_x_coords, list_y_coords]
