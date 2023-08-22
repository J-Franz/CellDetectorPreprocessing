import os
import zarr
import dask.array as da
from JonasTools.omero_tools import refresh_omero_session, get_image
from Dockerfile_build.Utils.utils import extract_system_arguments, unpack_parameters
import numpy as np
import nibabel as nib
import omero

def save_channel_to_nib_file(sys_arguments, parameters, channel=None):
    c_dapi, c_fluorescence, imageId, local, pw, user = extract_system_arguments(sys_arguments)
    if (channel == None):
        channel = c_dapi

    evaluated_crop_size, half_height, half_width, height, maximum_crop_size, overlap, width = unpack_parameters(
        parameters)

    with refresh_omero_session(None, user, pw) as conn:
        image = get_image(conn, imageId)
        group_id = image.getDetails().getGroup().getId()
        group_name = image.getDetails().getGroup().getName()
        conn.setGroupForSession(group_id)

        size_x = image.getSizeX()
        size_y = image.getSizeY()
        size_c = image.getSizeC()
        pixel_size_x =image.getPixelSizeX()
        pixel_size_y =image.getPixelSizeY()
        max_c = size_c - 1  # counting starts from 0

        roi_service = conn.getRoiService()

        rois_of_image = roi_service.findByImage(image.getId(), None)
        top_middle = False
        for roi in rois_of_image.rois:
            # print("ROI:  ID:", roi.getId().getValue())
            for s in roi.copyShapes():
                #print(type(s))
                if type(s) in (
                        omero.model.LabelI, omero.model.PointI):
                    if s.getTextValue():
                        text_value = s.getTextValue().getValue()
                        if text_value.find("top_middle") != -1:
                            top_middle = [s]
        if not top_middle:
            print("Top Middle not defined")
            return 0
    # Define channels automatically if not specified
    if c_dapi is None:
        c_dapi = 0
    if c_fluorescence is None:
        c_fluorescence = max_c

    if (channel == None):
        channel = c_dapi
    if local==True:
        zarr_path = "/home/jonas/SCRATCH2/Analysis/MicrogliaDepletionPreprocessed/Zarr/"
        nib_path = "/home/jonas/SCRATCH2/Analysis/MicrogliaDepletionPreprocessed/nib/"
        #zarr_path = "/share/Work/Neuropathologie/MicrogliaDetection/Playground/MicrogliaDepletionPreprocessed/Zarr/"
    else:
        zarr_path = "/scratch2/jfranz/Analysis/MicrogliaDepletionPreprocessed/Zarr/"
        nib_path = "/scratch2/jfranz/Analysis/MicrogliaDepletionPreprocessed/nib/"
    os.makedirs(zarr_path, exist_ok=True)
    os.makedirs(nib_path, exist_ok=True)
    print("Results are saved to directory: " + nib_path)



    fname = str(imageId) + "_image.zarr"
    for filename_ in os.listdir(zarr_path):

        if (filename_ == fname):
            z1 = zarr.open(zarr_path + fname, mode='r', shape=(size_x, size_y, size_c), chunks=(500, 500), dtype='i2')
            zarr_image = da.from_zarr(zarr_path + fname)
            # definition from B6_dapi project, pixel_size_x and pixel_size_y should be identical except for numerical deviations
            pixdim = ((pixel_size_x+pixel_size_y)/2.)/10**3


            resolution_ratio_to_B6_standard_image = (1/1.1673)/((pixel_size_x+pixel_size_y)/2.)  #

            shrink = 10*resolution_ratio_to_B6_standard_image # factor 10 comes from B6_dapi_project
            image_dim = (z1.shape[0], z1.shape[1])
            scale = 1 / shrink
            shrink = int(shrink)
            new_dim = (round(image_dim[1] * scale), round(image_dim[0] * scale))
            new_arr = np.ndarray(new_dim + (z1.shape[2],))
            print(f'Pixel dimensions: {pixdim} mm')

            cur_channel = zarr_image[::shrink, ::shrink, 0:2]

            percentile_lower = da.percentile(cur_channel[:,:,0].flatten(),6)

            percentile_upper = da.percentile(cur_channel[:,:,0].flatten(),80)
            cur_channel = cur_channel-percentile_lower
            cur_channel = cur_channel/(percentile_upper-percentile_lower)*255.
            cur_channel = cur_channel.clip(0,255)
            tm_x = top_middle[0].getX().getValue()
            tm_y = top_middle[0].getY().getValue()
            x_middle = (tm_x>size_x/4)&(tm_x<size_x*3/4)
            x_top = tm_x>size_x*3/4
            y_middle = (tm_y>size_y/4)&(tm_y<size_y*3/4)
            y_top = tm_y>size_y*3/4
            size_x_smaller_y = size_x < size_y

            if size_x_smaller_y:
                cur_channel = da.stack((cur_channel[:, :, 0].T, cur_channel[:, :, 1].T), 2)
                if x_top & y_middle:
                    cur_channel = da.flip(cur_channel,1)
            else:
                cur_channel = da.stack((cur_channel[:, :, 0], cur_channel[:, :, 1]), 2)
                if y_top & x_middle:
                    cur_channel = da.flip(cur_channel,1)


            cur_channel = da.array(cur_channel,dtype="uint32")
            nii_scaled = nib.Nifti1Image(cur_channel.compute(), np.eye(4))
            nii_scaled.header['pixdim'][1:3] = pixdim * shrink, pixdim * shrink
            save_name = str(nib_path)+filename_.split(".")[0]+".nii"
            nib.save(nii_scaled, save_name)

            return save_name, channel

            #image_data_to_nii(pixdim, output, shrink, out_dir, tif_path)
