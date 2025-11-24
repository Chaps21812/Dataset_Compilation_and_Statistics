from src.coco_tools import build_coco_satnet, train_test_split, build_raw_satnet, COCODataset
from src.preprocess_functions import raw_file_16bit, iqr_log_16bit
from src.raw_datset import raw_dataset

SATNET_DIRECTORY="/data/SatNet/SatNet/SatNet.v.1.4.0.0"
output_directory = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/SatNet1.4"

# build_coco_satnet(SATNET_DIRECTORY,output_directory,[raw_file_16bit, iqr_log_16bit])
# build_raw_satnet(SATNET_DIRECTORY,output_directory)
raw = raw_dataset(output_directory)
coco = COCODataset(output_directory)
coco.build_annotations()
coco.generate_TTV_split()
coco.move_fits_to_train_test_split(iqr_log_16bit)


