import os


def define_path(local):
    if local:
        path = ""
    else:
        path = "/scratch/users/jfranz/Analysis/Cellpose/MicrogliaDepletion/"
        os.makedirs(path, exist_ok=True)
        print("Results are saved to directory: " + path)
    return path

