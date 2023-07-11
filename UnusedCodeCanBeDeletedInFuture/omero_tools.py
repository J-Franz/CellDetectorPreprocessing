from collections import OrderedDict

import numpy as np

import os
from omero.gateway import BlitzGateway



def refresh_omero_session(conn,user,pw,verbose=False):
    if conn==None:
        USERNAME = user
        PASSWORD = pw
        HOST = "134.76.18.202"
        PORT=   4064
        if verbose:
            print("Connected.")
        conn = BlitzGateway(USERNAME, PASSWORD,host=HOST, port=PORT)
    else:
        
        USERNAME = user
        PASSWORD = pw
        HOST = "134.76.18.202"
        PORT=   4064

        conn.connect()
        if verbose:
            print("Connected.")
        conn = BlitzGateway(USERNAME, PASSWORD,host=HOST, port=PORT)
    conn.connect()
    if verbose:
        print(conn.isConnected())
    return conn


def get_pixels(conn, image, verbose = True):
    group_id = image.getDetails().getGroup().getId()
    conn.setGroupForSession(group_id)
    if verbose ==True:
        group_name = image.getDetails().getGroup().getName()
        print("Switched Group to Group of image: ", group_id, " with name: ", group_name)
    pixels = image.getPrimaryPixels()
    return pixels


def get_image(conn,  imageId):
    conn.SERVICE_OPTS.setOmeroGroup('-1')
    image = conn.getObject("Image", imageId)
    return image


def get_tile_coordinates(image, nx, ny, evaluated_crop_size, maximum_crop_size):
    current_crop_x = nx * evaluated_crop_size
    current_crop_y = ny * evaluated_crop_size
    crop_width = min(maximum_crop_size, int(image.getSizeX() - current_crop_x))
    crop_height = min(maximum_crop_size, int(image.getSizeY() - current_crop_y))
    return OrderedDict(current_crop_x=current_crop_x,
                       current_crop_y=current_crop_y,
                       crop_width=crop_width,
                       crop_height=crop_height)


def execute_command(command, verbose=False):
    with os.popen(command) as stream:
        output = stream.read()
        if verbose:
            print(output)
    return output

def UploadArrayAsTxtToOmero(fname, array, group_name, imageId, pw, user, verbose=True):
    np.savetxt(fname, array, delimiter=',', fmt='%f')
    # Upload file via omero client in bash system steered by python to the omero server and link to the image
    login_command = "omero login " + user + "@134.76.18.202 -w " + pw + " -g \"" + group_name + "\""
    execute_command(login_command)
    command = "omero upload " + fname
    output = execute_command(command)
    command = "omero obj new FileAnnotation file=" + output
    output = execute_command(command)
    command = "omero obj new ImageAnnotationLink parent=" + "Image:" + str(imageId) + " child=" + output
    return execute_command(command, verbose=verbose)


def check_fname_omero(fname, image):
    already_extracted = False
    for ann in image.listAnnotations():
        filename_ = ann.getFileName()
        if filename_ == fname:
            already_extracted = True
    return already_extracted