from collections import OrderedDict

import numpy as np

import os

import omero
from omero.gateway import BlitzGateway
import subprocess


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
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True
    )
    output = proc.stdout

    if verbose:
        print(proc.stderr)
        print(proc.stdout)
    return output

def UploadArrayAsTxtToOmero(fname, array, group_name, imageId, pw, user, verbose=True):
    if not os.path.isfile(fname):
        if type(array)!=type(None):
            np.savetxt(fname, array, delimiter=',', fmt='%f')
        else:
            print("We could neither find the file in the system nor did you provide an array.")
    # Upload file via omero client in bash system steered by python to the omero server and link to the image
    login_command = '''module load anaconda3
                    source activate /usr/users/jfranz/envs/cellpose_new
                    omero login {user} + @134.76.18.202 -w {pw} -g {groupname} '''.format(user=user, pw=pw, groupname=group_name)
    execute_command(login_command)
    command = '''module load anaconda3
                    source activate /usr/users/jfranz/envs/cellpose_new
                    omero upload {fname} '''.format(fname=fname)
    output = execute_command(command)
    command = '''module load anaconda3
                    source activate /usr/users/jfranz/envs/cellpose_new
                    omero obj new FileAnnotation file={fname} '''.format(fname=output)
    output = execute_command(command)
    command = '''module load anaconda3
                    source activate /usr/users/jfranz/envs/cellpose_new
                    omero obj new ImageAnnotationLink parent=Image:{Image} child={output}'''.format(Image=imageId, output=output)
    return execute_command(command, verbose=verbose)


def check_fname_omero(fname, image):
    already_extracted = False
    for ann in image.listAnnotations():
        if (type(ann)==omero.gateway.FileAnnotationWrapper):
            filename_ = ann.getFileName()
            if filename_ == fname:
                already_extracted = True
    return already_extracted


def make_omero_file_available(image, fname, path):
    # image: provide image object during active omero session
    # fname: string of the file name annotation to download
    # path: path where to save annotation

    if os.path.isfile(path+fname):
        print("File %s already saved in directory %s." %(fname, path))
        successful = True
    else:
        successful = download_fname_from_omero_image_annotations(fname, image, path)

    if successful == False:
        print("File neither in directory nor linked on omero.")
        return 1
    else:
        return 0


def download_fname_from_omero_image_annotations(fname, image, path):
    successful = False
    for ann in image.listAnnotations():
        if isinstance(ann, omero.gateway.FileAnnotationWrapper):
            if ann.getFile().getName() == fname:
                print("File ID:", ann.getFile().getId(), ann.getFile().getName(), \
                      "Size:", ann.getFile().getSize())
                file_path = os.path.join(path, ann.getFile().getName())

                with open(str(file_path), 'wb') as f:
                    print("\nDownloading file to", file_path, "...")
                    for chunk in ann.getFileInChunks():
                        f.write(chunk)
                print("File downloaded!")
                successful = True
    return successful