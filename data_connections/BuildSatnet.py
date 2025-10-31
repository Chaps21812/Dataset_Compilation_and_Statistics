from src.coco_tools import build_coco_satnet, train_test_split, set_primary_images_folder, set_ttv_primary_images_folder
from src.preprocess_functions import raw_file_16bit, iqr_log_16bit

SATNET_DIRECTORY="/data/SatNet/SatNet/SatNet.v.1.4.0.0"
output_directory = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/SatNet1.4"

build_coco_satnet(SATNET_DIRECTORY,output_directory,[raw_file_16bit, iqr_log_16bit])
set_primary_images_folder(output_directory,raw_file_16bit)
train_test_split(output_directory)
