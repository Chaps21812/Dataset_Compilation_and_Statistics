import zipfile
import random
import os
import json
import shutil
import datetime as dt
import numpy as np
from pandas_statistics import file_path_loader
from tqdm import tqdm

def compress_data(data_paths:list[str], output_zip:str) -> None:
    """
    Zips all images in the image path into a zipfile in the output path. Compresses each file one by one to reduce overhead memory requirements 

    Args:
        input_data (list[str]): Input list of files to compress
        stars (str): Output ZIP path, must end in abc.zip
    """
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for index, image_path in enumerate(data_paths):
            if index%500 == 0: print("{}/{}".format(index,len(data_paths)))
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
    for i, directory in enumerate(coco_directories):
        anotations_file = directory + "/annotations/annotations.json"
        with open(anotations_file, 'r') as f:
            annotations = json.load(f)
        images_queue.extend(annotations["images"])
        annotations_queue.extend(annotations["annotations"])
        notes_queue.append({"info":annotations["info"], "notes":annotations["notes"], "directory": directory })

        catagories_queue.extend(annotations["catagories"]) #try and find a smarter way of dealing with catagories pls

        templist = [directory+"/images/"+image for image in os.listdir(directory + "/images")]
        path_to_image.extend(templist)
    annotations_id_to_index = [[] for i in range(len(images_queue))]
    for index, item in enumerate(images_queue):
        id_to_index[item["id"]] = index
    for index,anot in enumerate(annotations_queue):
        annotations_id_to_index[id_to_index[anot["image_id"]]].append(index)
    
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
        titles = ["/train","/val","/test"]
        file_list_split = split_files(path_to_image, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        for j,temp_list in enumerate(file_list_split):
            indicies = [id_to_index[int(single_path.split("/")[-1].replace(".fits",""))] for single_path in temp_list]
            temp_anot_index = [annotations_id_to_index[id_to_index[int(single_path.split("/")[-1].replace(".fits",""))]] for single_path in temp_list]

            ### For one large dataset
            # Save annotation data to corresponding train test json
            #Make new data directory
            data_folder = destination_path
            images_alias = data_folder+titles[j]+"/images"
            annotations_alias = data_folder+titles[j]+"/annotations"
            if not os.path.exists(data_folder):os.mkdir(data_folder)
            if not os.path.exists(data_folder+titles[j]):os.mkdir(data_folder+titles[j])
            if not os.path.exists(images_alias):os.mkdir(images_alias)
            if not os.path.exists(annotations_alias):os.mkdir(annotations_alias)
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
                "catagories": catagories_queue,
                "notes": notes}
            data_attributes_obj=json.dumps(all_data, indent=4)
            with open("{}/annotations.json".format(annotations_alias), "w") as outfile:
                outfile.write(data_attributes_obj)

            #Compressing images without copying them to folder to save on memory
            if zip:
                print("Compressing images...")
                compress_data(temp_list, images_alias+"/compressed_train_fits.zip")
            else:
                total_copies = len(temp_list)
                for i,image in enumerate(temp_list):
                    if i%100==0:print(f"{i}/{total_copies}")
                    shutil.copy(image, images_alias)
    else:
        ### For one large dataset
        # Save annotation data to corresponding train test json
        #Make new data directory
        data_folder = destination_path
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
            os.mkdir(data_folder+"/images")
            os.mkdir(data_folder+"/annotations")

        all_data = {
            "info": info,
            "images": images_queue,
            "annotations": annotations_queue,
            "catagories": catagories_queue,
            "notes": notes}
        data_attributes_obj=json.dumps(all_data, indent=4)
        with open("{}/annotations/annotations.json".format(data_folder), "w") as outfile:
            outfile.write(data_attributes_obj)

        #Compressing images without copying them to folder to save on memory
        if zip:
            print("Compressing images...")
            compress_data(path_to_image, data_folder+"/images/compressed_fits.zip")
        else:
            total_copies = len(path_to_image)
            for i,image in enumerate(path_to_image):
                if i%100==0:print(f"{i}/{total_copies}")
                shutil.copy(image, data_folder+"/images")

def silt_to_coco(silt_dataset_path:str, zip:bool=False, notes:str=""):
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
    loader =  file_path_loader(silt_dataset_path)
    satsim_paths = os.listdir(satsim_path)

    for i,path in tqdm(enumerate(satsim_paths), desc="Converting Silt to COCO"):
        raw_fits_path=os.path.join(satsim_path, path, "raw_fits")
        raw_annotations_path=os.path.join(satsim_path, path, "raw_annotation")
        fits_path=os.path.join(satsim_path, path, "ImageFiles")
        annotations_path=os.path.join(satsim_path, path, "Annotations")

        fits_path=satsim_path+"/"+path+"/ImageFiles"
        annotations_path=satsim_path+"/"+path+"/Annotations"
        config_path = satsim_path+"/"+path+"/config.json"

        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        for frame_no, fits_name in enumerate(os.listdir(fits_path)):
            #Path and filename manipulation
            fits_file = fits_path+"/"+fits_name
            json_file = annotations_path +"/"+ fits_name.replace(".fits", ".json")
            with open(json_file, 'r') as f:
                #Load satellite annotation data
                json_data = json.load(f)
                object_list = json_data["data"]["objects"]
                data = json_data["data"]
                image_id= np.random.randint(0,9223372036854775806)
                annotations = []
                #Process all detected objects
                for object in object_list:
                    if not object["class_id"] == 2: continue
                    if 0>object["x_start"]: object["x_start"]=0 
                    if 0>object["x_end"]: object["x_end"]=0 
                    if 0>object["y_start"]: object["y_start"]=0 
                    if 0>object["y_end"]: object["y_end"]=0 
                    if 1<object["x_start"]: object["x_start"]=1 
                    if 1<object["x_end"]: object["x_end"]=1 
                    if 1<object["y_start"]: object["y_start"]=1 
                    if 1<object["y_end"]: object["y_end"]=1 
                    if 0>object["y_min"]: object["y_min"]=0 
                    if 0>object["x_min"]: object["x_min"]=0 
                    if 0>object["x_max"]: object["x_max"]=0 
                    if 0>object["y_max"]: object["y_max"]=0 
                    if 1<object["y_min"]: object["y_min"]=1 
                    if 1<object["x_min"]: object["x_min"]=1 
                    if 1<object["x_max"]: object["x_max"]=1 
                    if 1<object["y_max"]: object["y_max"]=1 

                    #Create coco annotation for one image
                    annotation = {
                        "id": np.random.randint(0,9223372036854775806),
                        "image_id": image_id,
                        "category_id": object["class_id"]-1,
                        "type": "bbox",
                        "centroid": [object["x_center"],object["y_center"]],
                        "bbox": [object["x_center"],object["y_center"],object["bbox_width"],object["bbox_height"]],
                        "area": object["bbox_width"]*object["bbox_height"],
                        "line": [object["x_start"],object["y_start"],object["x_end"],object["y_end"]],
                        "line_center": [object["x_mid"],object["y_mid"]],
                        "magnitude": object["magnitude"],
                        "ra":object["ra"],
                        "dec": object["dec"],
                        "iscrowd": 0,
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
                    "file_name": str(image_id)+".fits",
                    "original_path": fits_file,
                    "frame_no": frame_no,
                    "date": config_data["geometry"]["time"]}

                #Add coco image to list of files
                path_to_annotation[fits_file] = {"annotation":annotations, "image":image, "new_id":image_id}

    #Compile final json information for folder
    category1 = {"id": 1-1,"name": "Satellite","supercategory": "Space Object",}
    category2 = {"id": 2-1,"name": "Star","supercategory": "Space Object",}
    catagories = [category1, category2]
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
    data_folder = destination_path
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
        os.mkdir(data_folder+"/images")
        os.mkdir(data_folder+"/annotations")

    images = []
    annotations = []
    for path in path_to_annotation.keys():
        image = path_to_annotation[path]["image"]
        annotation = path_to_annotation[path]["annotation"]
        images.append(image)
        annotations.extend(annotation)
    all_data = {
        "info": info,
        "images": images,
        "annotations": annotations,
        "catagories": catagories,
        "notes": notes}
    data_attributes_obj=json.dumps(all_data, indent=4)
    with open("{}/annotations/annotations.json".format(data_folder), "w") as outfile:
        outfile.write(data_attributes_obj)

    #Compressing images without copying them to folder to save on memory
    if zip:
        print("Compressing all images...")
        compress_data(path_to_annotation.keys(), data_folder+"/images/compressed_train_fits.zip")
    else:
        total_copies = len(path_to_annotation.keys())
        for i,image in enumerate(path_to_annotation.keys()):
            filename = image.split("/")[-1]
            if i%100==0:print(f"{i}/{total_copies}")
            shutil.copy(image, data_folder+"/images")
            shutil.move(data_folder+"/images/"+filename,data_folder+"/images/"+str(path_to_annotation[image]["new_id"])+".fits" )