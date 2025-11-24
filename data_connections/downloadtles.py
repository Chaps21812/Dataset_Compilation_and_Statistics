from src.raw_datset import raw_dataset
from src.coco_tools import COCODataset
from src.preprocess_functions import raw_file, iqr_log
from src.s3client import S3Client
import os

# Please just download the fucking TLES bro. 

if __name__ == "__main__":
    
    from src.raw_datset import raw_dataset
    from src.coco_tools import COCODataset
    import os

    telescope_directory = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/CalsatTesting_datasets"

    # dataset = raw_dataset("/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/CalsatTesting_datasets/Calsats-RME04-2024").download_TLEs()


    telescope_basename = os.path.basename(telescope_directory)
    for date_folder in os.listdir(telescope_directory):
        subfolder = os.path.join(telescope_directory, date_folder)
        if os.path.isdir(subfolder):
            dataset = raw_dataset(subfolder).download_TLEs()


