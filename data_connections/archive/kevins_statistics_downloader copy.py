import boto3
from statistics_obj import Statistics_obj
import json
import io
from astropy.io import fits
import os
from datetime import datetime
import numpy as np
import random
from PIL import Image
from astropy.visualization import ZScaleInterval
from tqdm import tqdm

# Local folders
os.makedirs("RME04v4_annotations", exist_ok=True)
os.makedirs("RME04v4_fits_files", exist_ok=True)
os.makedirs("RME04v4_preprocessed_images", exist_ok=True)

# To randomly go through the annoations instead of the first on the list
random.shuffle(annotations)

for i, annot in enumerate(tqdm(annotations[:1000], desc="Processing annotations")):
    try:
        response = client.get_object(Bucket=bucket, Key=annot)
        json_content = json.loads(response['Body'].read().decode('utf-8'))
        filename = json_content["file"]["filename"]
        fits_file = os.path.join(os.path.dirname(annot), "ImageFiles", filename)
        response = client.get_object(Bucket=bucket, Key=fits_file)
        fits_content = io.BytesIO(response['Body'].read())

        with fits.open(fits_content) as hdul:
            image_data = hdul[0].data
            hdu = hdul[0].header
            tm = hdu["TRKMODE"]
            # Save FITS locally
            if tm == 'rate':
                fits_local_path = os.path.join("RME04v4_fits_files", filename)
                with open(fits_local_path, "wb") as f:
                    f.write(fits_content.getbuffer())
        
        # Save JSON locally
        if tm == 'rate':
            json_filename = os.path.basename(annot)
            with open(os.path.join("RME04v4_annotations", json_filename), "w") as f:
                json.dump(json_content, f, indent=4)
       
        # Save as 16-bit PNG
        if tm == 'rate':
            png_filename = os.path.splitext(filename)[0] + ".png"
            Image.fromarray(image_data_png).save(os.path.join("RME04v4_preprocessed_images", png_filename))



