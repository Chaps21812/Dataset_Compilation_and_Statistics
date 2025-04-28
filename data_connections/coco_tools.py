import zipfile
import random
import os
import json
import shutil
import datetime as dt
import numpy as np
from pandas_statistics import file_path_loader
from tqdm import tqdm
from astropy.io import fits
import pycocotools

def merge_categories(category_list:list):
    name_to_category = {}
    
    for cat in category_list:
        name = cat['name']
        if name not in name_to_category:
            name_to_category[name] = {
                "id": cat["id"],
                "name": cat['name'],
                "supercategory": cat.get("supercategory", "")
            }

    return list(name_to_category.values())

def compress_data(data_paths:list[str], output_zip:str) -> None:
    """
    Zips all images in the image path into a zipfile in the output path. Compresses each file one by one to reduce overhead memory requirements 

    Args:
        input_data (list[str]): Input list of files to compress
        stars (str): Output ZIP path, must end in abc.zip
    """
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for image_path in tqdm(data_paths, desc="Compressing images", total=len(data_paths)):
            zipf.write(image_path, arcname=os.path.basename(image_path))  # Save only filename

def split_files(file_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15) -> tuple:
    """
    Generates a train test split given a list of files. 

    Args:
        input_data (list[str]): Input list of files to shuffle and generate a train test split
        train_ratio (float): Ratio of training samples
        val_ratio (float): Ratio of validation samples
        test_ratio (float): Ratio of testing samples

    Returns:
        train, test, split (tuple): List of files present in the train test split
    """

    # Ensure the ratios add up to 1
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("The sum of train, val, and test ratios must be 1.")
    
    # Shuffle the file list to ensure random distribution
    random.shuffle(file_list)
    
    # Calculate the split sizes
    total_files = len(file_list)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)
    test_size = total_files - train_size - val_size  # The remainder will be the test set

    # Split the files
    train_files = file_list[:train_size]
    val_files = file_list[train_size:train_size + val_size]
    test_files = file_list[train_size + val_size:]

    return train_files, val_files, test_files

def merge_coco(coco_directories:list[str], destination_path:str, notes:str="", train_test_split:bool=False, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, zip:bool=False):
    path_to_image = []
    images_queue = []
    annotations_queue = []
    notes_queue = []
    catagories_queue = []
    id_to_index = {}
    for directory in tqdm(coco_directories, desc="Processing COCO Datasets"):
        anotations_file = os.path.join(directory, "annotations", "annotations.json")
        with open(anotations_file, 'r') as f:
            annotations = json.load(f)
        images_queue.extend(annotations["images"])
        annotations_queue.extend(annotations["annotations"])
        notes_queue.append({"info":annotations["info"], "notes":annotations["notes"], "directory": directory })

        catagories_queue.extend(annotations["categories"]) #try and find a smarter way of dealing with categories pls
        templist = [os.path.join(directory, "images", image) for image in os.listdir(os.path.join(directory, "images"))]
        path_to_image.extend(templist)
    annotations_id_to_index = [[] for i in range(len(images_queue))]
    for index, item in enumerate(images_queue):
        id_to_index[item["id"]] = index
    for index,anot in enumerate(annotations_queue):
        annotations_id_to_index[id_to_index[anot["image_id"]]].append(index)
    merged_catagories = merge_categories(catagories_queue)

    
    now = dt.datetime.now()
    info = {
        "year": now.year,
        "version": "1.0",
        "description": "Satellite detection of calsat dataset. ",
        "contributor":"EO Solutions",
        "date_created": now.strftime("%Y-%m-%d %H:%M:%S"),
        "annotation": "Satellite BBox",
        "samples":len(path_to_image),
        "prev_notes":notes_queue}
    
    if train_test_split:
        titles = ["train","val","test"]
        file_list_split = split_files(path_to_image, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        for j,temp_list in enumerate(file_list_split):
            indicies = [id_to_index[int(single_path.split("/")[-1].replace(".fits",""))] for single_path in temp_list]
            temp_anot_index = [annotations_id_to_index[id_to_index[int(single_path.split("/")[-1].replace(".fits",""))]] for single_path in temp_list]

            ### For one large dataset
            # Save annotation data to corresponding train test json
            #Make new data directory
            data_folder = destination_path
            subset_folder = os.path.join(data_folder, titles[j])
            images_alias = os.path.join(data_folder, titles[j], "images")
            annotations_alias = os.path.join(data_folder, titles[j], "annotations")
            os.makedirs(data_folder, exist_ok=True)
            os.makedirs(subset_folder, exist_ok=True)
            os.makedirs(images_alias, exist_ok=True)
            os.makedirs(annotations_alias, exist_ok=True)

            info["train"]= train_ratio
            info["test"]= test_ratio
            info["val"]= val_ratio
            info["samples"]=len(temp_list)

            t_anot =  [[annotations_queue[j] for j in li ] for li in temp_anot_index]
            all_anot = []
            for t in t_anot:
                all_anot.extend(t)

            all_data = {
                "info": info,
                "images": [images_queue[i] for i in indicies],
                "annotations":all_anot,
                "categories": merged_catagories,
                "notes": notes}
            data_attributes_obj=json.dumps(all_data, indent=4)
            with open(os.path.join(annotations_alias, "annotations.json"), "w") as outfile:
                outfile.write(data_attributes_obj)

            
            #Compressing images without copying them to folder to save on memory
            if zip:
                compress_data(temp_list, os.path.join(images_alias, "compressed_fits.zip"))
            else:
                for image in tqdm(temp_list, desc="Copying images", total=len(temp_list)):
                    shutil.copy(image, images_alias)
    else:
        ### For one large dataset
        # Save annotation data to corresponding train test json
        #Make new data directory
        data_folder = destination_path
        new_images_folder = os.path.join(data_folder, "images")
        new_annotations_folder = os.path.join(data_folder, "annotations")
        os.makedirs(data_folder, exist_ok=True)
        os.makedirs(new_images_folder, exist_ok=True)
        os.makedirs(new_annotations_folder, exist_ok=True)

        all_data = {
            "info": info,
            "images": images_queue,
            "annotations": annotations_queue,
            "categories": catagories_queue,
            "notes": notes}
        data_attributes_obj=json.dumps(all_data, indent=4)
        with open(os.path.join(new_annotations_folder, "annotations.json"), "w") as outfile:
            outfile.write(data_attributes_obj)

        #Compressing images without copying them to folder to save on memory
        if zip:
            compress_data(path_to_image, os.path.join(new_images_folder, "compressed_fits.zip"))
        else:
            for image in tqdm(path_to_image, desc="Copying images", total=len(path_to_image)):
                shutil.copy(image, new_images_folder)

def silt_to_coco(silt_dataset_path:str, include_sats:bool=True, include_stars:bool=False, zip:bool=False, notes:str=""):
    """
    Converts a satasim generated dataset into a coco dataset

    Args:
        path (list[str]): Input list of files to shuffle and generate a train test split
        train_ratio (float): Ratio of training samples
        val_ratio (float): Ratio of validation samples
        test_ratio (float): Ratio of testing samples

    Returns:
        train, test, split (tuple): List of files present in the train test split
    """
    path_to_annotation = {}
    local_files =  file_path_loader(silt_dataset_path)

    for annotation_path,fits_path in tqdm(local_files.annotation_to_fits.items(), desc="Converting Silt to COCO"):        
        with open(annotation_path, 'r') as f:
            json_data = json.load(f)
        hdu = fits.open(fits_path)
        hdul = hdu[0]
        header = hdul.header
        object_list = json_data["objects"]
        image_id= np.random.randint(0,9223372036854775806)
        annotations = []
        #Process all detected objects
        for object in object_list:
            if include_sats and object["class_name"] == "Satellite": 
                #Create coco annotation for one image
                annotation = {
                    "id": np.random.randint(0,9223372036854775806),
                    "image_id": image_id,
                    "category_id": object["class_id"],# Originallly class_id-1, not sure why
                    "category_name": object["class_name"],
                    "type": "bbox",
                    "centroid": [object["x_center"],object["y_center"]],
                    "bbox": [object["x_center"],object["y_center"],object["bbox_width"],object["bbox_height"]],
                    "area": object["bbox_width"]*object["bbox_height"],
                    "line": [object["x_start"],object["y_start"],object["x_end"],object["y_end"]],
                    "line_center": [object["x_mid"],object["y_mid"]],
                    "iso_flux": object["iso_flux"],
                    "iscrowd": 0,
                    "y_min": object["y_center"]-object["bbox_height"]/2,
                    "x_min": object["x_center"]-object["bbox_width"]/2,
                    "y_max": object["y_center"]+object["bbox_height"]/2,
                    "x_max": object["x_center"]+object["bbox_width"]/2,
                    }
                annotations.append(annotation)

            if include_stars and object["class_name"] == "Star": 
                #Create coco annotation for one image
                width = abs(object["x1"] - object["x2"])
                height = abs(object["y1"] - object["y2"])
                annotation = {
                    "id": np.random.randint(0,9223372036854775806),
                    "image_id": image_id,
                    "category_id": object["class_id"],# Originallly class_id-1, not sure why
                    "category_name": object["class_name"],
                    "type": "bbox",
                    "centroid": [object["x_center"],object["y_center"]],
                    "bbox": [object["x_center"],object["y_center"],width,height],
                    "area": width*height,
                    "line": [object["x1"],object["y1"],object["x2"],object["y2"]],
                    "line_center": [object["x_center"],object["y_center"]],
                    "iso_flux": object["iso_flux"],
                    "iscrowd": 0,
                    }
                annotations.append(annotation)

        
        image = {
            "id": image_id,
            "width": json_data["sensor"]["width"],
            "height": json_data["sensor"]["height"],
            "type":"siderial" if header["TELTKRA"] == 0.0 else "rate",
            "rate": header["TELTKRA"],
            "exposure_seconds":header["EXPTIME"],
            "gain":1,
            "lat":header["SITELAT"],
            "lat":header["SITELAT"],
            "lon":header["CENTALT"],
            "file_name": os.path.join("images", f"{image_id}.fits"),
            "original_path": fits_path,
            "date": header["DATE-OBS"]}

        #Add coco image to list of files
        path_to_annotation[fits_path] = {"annotation":annotations, "image":image, "new_id":image_id}

    #Compile final json information for folder
    category1 = {"id": 0,"name": "Satellite","supercategory": "Space Object",}
    category2 = {"id": 1,"name": "Star","supercategory": "Space Object",}
    categories = [category1, category2]
    now = dt.datetime.now()
    info = {
        "year": now.year,
        "version": "1.0",
        "description": notes,
        "contributor":"EO Solutions",
        "date_created": now.strftime("%Y-%m-%d %H:%M:%S")}


    ### For one large dataset
    #Make new data directory
    new_fits_path=os.path.join(silt_dataset_path, "images")
    new_annotations_path=os.path.join(silt_dataset_path, "annotations")
    os.makedirs(new_fits_path, exist_ok=True)
    os.makedirs(new_annotations_path, exist_ok=True)


    images = []
    annotations = []
    for path in path_to_annotation.keys():
        images.append(path_to_annotation[path]["image"])
        annotations.extend(path_to_annotation[path]["annotation"])
        
    #Writing annotations to the json annotations file
    all_data = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "notes": notes}
    with open(os.path.join(new_annotations_path, "annotations.json"), "w") as outfile:
        data_attributes_obj=json.dumps(all_data, indent=4)
        outfile.write(data_attributes_obj)

    #Saving Images with or without compression
    if zip:
        compress_data(path_to_annotation.keys(), os.path.join(new_fits_path,"compressed_fits.zip"))
    else:
        for image in tqdm(path_to_annotation.keys(), desc="Copying images", total=len(path_to_annotation.keys())):
            new_fits_name = os.path.join(new_fits_path,str(path_to_annotation[image]["new_id"])+".fits")
            destination_path = shutil.copy(image, new_fits_path)
            shutil.move(destination_path,new_fits_name)

def satsim_to_coco(satsim_path:str, include_sats:bool=True, include_stars:bool=False, zip:bool=False, notes:str=""):
    """
    Converts a satasim generated dataset into a coco dataset

    Args:
        path (list[str]): Input list of files to shuffle and generate a train test split
        train_ratio (float): Ratio of training samples
        val_ratio (float): Ratio of validation samples
        test_ratio (float): Ratio of testing samples

    Returns:
        train, test, split (tuple): List of files present in the train test split
    """
    path_to_annotation = {}
    satsim_paths = os.listdir(satsim_path)

    for path in tqdm(satsim_paths, desc="Converting Satsim to COCO"):
        
        fits_path = os.path.join(satsim_path, path, "ImageFiles")
        annotations_path =  os.path.join(satsim_path, path, "Annotations")
        config_path = os.path.join(satsim_path, path, "config.json")
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        for frame_no, fits_name in enumerate(os.listdir(fits_path)):
            #Path and filename manipulation
            fits_file = os.path.join(fits_path, fits_name)
            json_file = os.path.join(annotations_path, fits_name.replace(".fits", ".json"))
            with open(json_file, 'r') as f:
                #Load satellite annotation data
                json_data = json.load(f)
                object_list = json_data["data"]["objects"]
                data = json_data["data"]
                image_id= np.random.randint(0,9223372036854775806)
                annotations = []
                #Process all detected objects
                for object in object_list:
                    if include_sats and object["class_name"] == "Satellite": 
                        #Create coco annotation for one image
                        annotation = {
                            "id": np.random.randint(0,9223372036854775806),
                            "image_id": image_id,
                            "category_id": object["class_id"],# Originallly class_id-1, not sure why
                            "category_name": object["class_name"],
                            "type": "bbox",
                            "centroid": [object["x_center"],object["y_center"]],
                            "bbox": [object["x_center"],object["y_center"],object["bbox_width"],object["bbox_height"]],
                            "area": object["bbox_width"]*object["bbox_height"],
                            "iscrowd": 0,
                            "snr": object["snr"],
                            "y_min": object["y_min"],
                            "x_min": object["x_min"],
                            "y_max": object["y_max"],
                            "x_max": object["x_max"],
                            "source": object["source"],
                            "magnitude": object["magnitude"],
                            }
                        annotations.append(annotation)

                    if include_stars and object["class_name"] == "Star": 
                        #Create coco annotation for one image
                        annotation = {
                            "id": np.random.randint(0,9223372036854775806),
                            "image_id": image_id,
                            "category_id": object["class_id"],# Originallly class_id-1, not sure why
                            "category_name": object["class_name"],
                            "type": "line",
                            "centroid": [object["x_center"],object["y_center"]],
                            "bbox": [object["x_center"],object["y_center"],object["bbox_width"],object["bbox_height"]],
                            "area": object["bbox_width"]*object["bbox_height"],
                            "line": [object["x_start"],object["y_start"],object["x_end"],object["y_end"]],
                            "line_center": [object["x_mid"],object["y_mid"]],
                            "iscrowd": 0,
                            "y_min": object["y_min"],
                            "x_min": object["x_min"],
                            "y_max": object["y_max"],
                            "x_max": object["x_max"],
                            "source": object["source"],
                            "magnitude": object["magnitude"],
                            }
                        annotations.append(annotation)

                image = {
                    "id": image_id,
                    "width": data["sensor"]["width"],
                    "height": data["sensor"]["height"],
                    "y_fov": config_data["fpa"]["y_fov"],
                    "x_fov": config_data["fpa"]["x_fov"],
                    "type":config_data["geometry"]["site"]["track"]["mode"],
                    "exposure_seconds":config_data["fpa"]["time"]["exposure"],
                    "gain":1,
                    "alt":config_data["geometry"]["site"]["alt"],
                    "lat":config_data["geometry"]["site"]["lat"],
                    "lon":config_data["geometry"]["site"]["lon"],
                    "file_name": os.path.join("images", f"{image_id}.fits"),
                    "original_path": fits_file,
                    "frame_no": frame_no,
                    "date": config_data["geometry"]["time"]}

                #Add coco image to list of files
                path_to_annotation[fits_file] = {"annotation":annotations, "image":image, "new_id":image_id}

    #Compile final json information for folder
    category1 = {"id": 0,"name": "Satellite","supercategory": "Space Object",}
    category2 = {"id": 1,"name": "Star","supercategory": "Space Object",}
    categories = [category1, category2]
    now = dt.datetime.now()
    info = {
        "year": now.year,
        "version": "1.0",
        "description": "Satellite detection of calsat dataset. ",
        "contributor":"EO Solutions",
        "date_created": now.strftime("%Y-%m-%d %H:%M:%S"),
        "annotation": "Satellite BBox"}


    ### For one large dataset
    # Save annotation data to corresponding train test json
    #Make new data directory
    data_folder = satsim_path
    images_folder = os.path.join(data_folder, "images")
    annotations_folder = os.path.join(data_folder, "annotations")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(annotations_folder, exist_ok=True)

    images = []
    annotations = []
    for path in path_to_annotation.keys():
        images.append(path_to_annotation[path]["image"])
        annotations.extend(path_to_annotation[path]["annotation"])
    all_data = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "notes": notes}
    data_attributes_obj=json.dumps(all_data, indent=4)
    with open(os.path.join(annotations_folder, "annotations.json"), "w") as outfile:
        outfile.write(data_attributes_obj)

    #Compressing images without copying them to folder to save on memory
    if zip:
        compress_data(path_to_annotation.keys(), os.path.join(images_folder, "compressed_fits.zip"))
    else:
        for image in tqdm(path_to_annotation.keys(), desc="Copying images", total=len(path_to_annotation.keys())):
            basename = os.path.basename(image)
            shutil.copy(image, images_folder)
            shutil.move(os.path.join(images_folder, basename), os.path.join(images_folder, str(path_to_annotation[image]["new_id"])+".fits") )