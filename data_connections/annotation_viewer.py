from astropy.io import fits
from pandas_statistics import file_path_loader
import os
import json
from plots import plot_image_with_bbox, plot_image_with_line, z_scale_image, plot_image, plot_all_annotations
from tqdm import tqdm

#Enter the dataset directory you wish to plot annotations for
dataset_path = "/mnt/c/Users/david.chaparro/Documents/Repos/Dataset-Statistics/data/TakoTruckSatellite"
view_satellite=False
view_star=False
view_image=True




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

