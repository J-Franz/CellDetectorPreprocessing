from collections import OrderedDict
import subprocess

import numpy as np

import os

import omero
from omero.gateway import BlitzGateway
import omero.gateway




def refresh_omero_session(conn,user,pw,verbose=False, instance="134.76.18.202"):
    if conn==None:
        USERNAME = user
        PASSWORD = pw
        HOST = instance
        PORT=   4064
        if verbose:
            print("Connected.")
        c = omero.client(host=HOST, port=PORT, args=[
            '--IceSSL.Ciphers=HIGH'])
        session = c.createSession(USERNAME, PASSWORD,  )
        conn = BlitzGateway(client_obj=c)
    else:
        # does not work currently
        USERNAME = user
        PASSWORD = pw
        HOST = f"wss://{instance}"
        PORT=   4064

        conn.connect()
        if verbose:
            print("Connected.")
        c = omero.client(host=HOST, port=PORT, args=[
            '--IceSSL.Ciphers=HIGH'])
        session = c.createSession(USERNAME, PASSWORD)

        conn = BlitzGateway(client_obj=c)
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
    try:
        # Execute the command
        result = subprocess.run(
            command,
            shell=True,  # Run the command through the shell
            check=True,  # Raise an exception if the command fails
            stdout=subprocess.PIPE,  # Capture standard output
            stderr=subprocess.PIPE,  # Capture standard error
            text=True  # Decode output as text
        )

        # If verbose, print the output and errors
        if verbose:
            print(result.stdout)
            if result.stderr:
                print("Error Output:", result.stderr)

        return result.stdout.strip()

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the command: {command}")
        print(f"Return Code: {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        raise  # Re-raise the exception after logging

def UploadArrayAsTxtToOmero(fname, array, group_name, imageId, pw, user, verbose=True, instance="134.76.18.202"):
    np.savetxt(fname, array, delimiter=',', fmt='%f')
    # Upload file via omero client in bash system steered by python to the omero server and link to the image
    login_command = "omero login " + user + f"@{instance} -w " + pw + " -g \"" + group_name + "\""
    try:
        execute_command(login_command, verbose=verbose)
    except:
        try:
            UploadArrayAsTxtToOmero(fname, array, group_name, imageId, pw, user, verbose=True)
            return False
        except:
            print("Command line upload and Python API Upload failed")
    command = "omero upload " + fname
    output = execute_command(command)
    command = "omero obj new FileAnnotation file=" + output
    output = execute_command(command)
    command = "omero obj new ImageAnnotationLink parent=" + "Image:" + str(imageId) + " child=" + output
    try:
        execute_command(command, verbose=verbose)
    except:
        try:
            print("Command line upload failed.")
            # TODO eventually replace
            #UploadArrayAsTxtToOmero_API(fname, array, group_name, imageId, pw, user, verbose=True)
        except:
            print("Command line upload and Python API Upload failed")

    return True

def check_fname_omero(fname, image):
    already_extracted = False
    for ann in image.listAnnotations():
        if ann.OMERO_TYPE == omero.model.FileAnnotationI:
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