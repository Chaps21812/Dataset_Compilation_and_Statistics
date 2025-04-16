from astropy.io import fits
from pandas_statistics import file_path_loader
import os
import json
from plots import plot_image_with_bbox, plot_image_with_line, z_scale_image, plot_image, plot_all_annotations
from tqdm import tqdm
import pandas as pd
import numpy as np
#Enter the dataset directory you wish to plot annotations for
dataset_path = "/mnt/c/Users/david.chaparro/Documents/Repos/Dataset-Statistics/data/RME03Star/"
view_satellite=False
view_star=False
view_image=True

def plot_annotation_subset(pandas_library:pd.DataFrame, dataset_path:str, view_satellite:bool=False, view_star:bool=False, view_image:bool=True):
    """
    Plots annotations from a dataset.

    Parameters:
    dataset_path (str): The path to the dataset directory.
    view_satellite (bool): Whether to plot satellite annotations.
    view_star (bool): Whether to plot star annotations.
    view_image (bool): Whether to plot the image with annotations.
    """

    loader = file_path_loader(dataset_path)
    annotation_view_path = os.path.join(dataset_path, "annotation_view")
    os.makedirs(annotation_view_path, exist_ok=True)



    for json_path in tqdm(pandas_library["json_path"].unique(), desc="Plotting annotations", unit="file"):
        fits_path = loader.annotation_to_fits[json_path]
        with open(json_path, 'r') as file:
            annotation = json.load(file)
        fits_file = fits.open(fits_path)
        hdu = fits_file[0].header
        raw_data = fits_file[0].data

        #The XY coordinates are reverse intentionally. Beware!
        y_res = hdu["NAXIS2"]
        x_res = hdu["NAXIS1"]

        noise = np.std(raw_data)
        median_pixel = np.median(raw_data)

        data = z_scale_image(raw_data)
        if view_image: plot_all_annotations(data, annotation["objects"], (x_res,y_res), json_path, dpi=500)
        for index,object in enumerate(annotation["objects"]):
            x_cord= object["x_center"]*x_res
            y_cord= object["y_center"]*y_res

            half = 50
            x_start = max(0, x_cord - half)
            x_end   = min(data.shape[1], x_cord + half)
            y_start = max(0, y_cord - half)
            y_end   = min(data.shape[0], y_cord + half)

            window = raw_data[int(y_start):int(y_end), int(x_start):int(x_end)]
            local_minimum = np.min(window)
            local_std = np.std(window)
            signal = raw_data[int(y_cord), int(x_cord)]
            snr_tuple = (object["x_center"]*x_res, object["y_center"]*y_res,"Prom: {}".format((signal-local_minimum)/local_std))

            if object['class_name']=="Satellite": 
                if view_satellite: plot_image_with_bbox(data,object['x_center']*x_res,object['y_center']*y_res,object['x_max']*x_res-object['x_min']*x_res,index, json_path, dpi=500, snr=snr_tuple)

            if object['class_name']=="Star": 
                if view_star: plot_image_with_line(data,object['x1']*x_res,object['y1']*y_res,object['x2']*x_res,object['y2']*y_res,index, json_path, dpi=500, snr=snr_tuple)

def plot_annotations(dataset_path:str, view_satellite:bool=False, view_star:bool=False, view_image:bool=True):
    """
    Plots annotations from a dataset.

    Parameters:
    dataset_path (str): The path to the dataset directory.
    view_satellite (bool): Whether to plot satellite annotations.
    view_star (bool): Whether to plot star annotations.
    view_image (bool): Whether to plot the image with annotations.
    """

    loader = file_path_loader(dataset_path)
    annotation_view_path = os.path.join(dataset_path, "annotation_view")
    os.makedirs(annotation_view_path, exist_ok=True)

    for json_path,fits_path in tqdm(loader.annotation_to_fits.items(), desc="Plotting annotations", unit="file"):
        with open(json_path, 'r') as file:
            annotation = json.load(file)
        fits_file = fits.open(fits_path)
        hdu = fits_file[0].header
        data = fits_file[0].data

        #The XY coordinates are reverse intentionally. Beware!
        x_res = hdu["NAXIS2"]
        y_res = hdu["NAXIS1"]

        data = z_scale_image(data)
        if view_image: plot_all_annotations(data, annotation["objects"], (x_res,y_res), json_path, dpi=500)
        for index,object in enumerate(annotation["objects"]):
            if object['class_name']=="Satellite": 
                if view_satellite: plot_image_with_bbox(data,object['x_center']*x_res,object['y_center']*y_res,object['x_max']*x_res-object['x_min']*x_res,index, json_path, dpi=500)

            if object['class_name']=="Star": 
                if view_star: plot_image_with_line(data,object['x1']*x_res,object['y1']*y_res,object['x2']*x_res,object['y2']*y_res,index, json_path, dpi=500)

