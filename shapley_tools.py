import numpy as np
from shapely import geometry
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib.path import Path


def get_polygon_as_shape(list_):
    polygon_list_ = []
    for polygon_ in list_:
        if len(polygon_)>0:
            PolygonOmero = polygon_
            omero_points_polygon = PolygonOmero[0].getPoints()
            points_string = omero_points_polygon.getValue().split()
            coords = []
            for point in points_string:
                point = point.split(',')
                coords.append((float(point[0]),float(point[1])))
            polygon_list_.append(geometry.Polygon(coords)) # Erzeuge Polygon aus Omero daten
        else:
            polygon_list_.append(None)
    return polygon_list_

def analyse_polygon_histogram(polygon,tile):

    # calculate boolean mask of encircled area of polygon
    polygon_path = np.vstack((polygon.exterior.xy[0], polygon.exterior.xy[1])).T
    polygon_path_matplot = Path(polygon_path)
    crop_width = np.shape(tile)[0]
    crop_height = np.shape(tile)[1]
    x, y = np.meshgrid(np.arange(crop_width), np.arange(crop_height))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    grid = polygon_path_matplot.contains_points(points)
    grid = grid.reshape((crop_height, crop_width))
    # calculate ECDF of median filtered image within polygon (defined by boolean grid)
    return ECDF(tile[grid])