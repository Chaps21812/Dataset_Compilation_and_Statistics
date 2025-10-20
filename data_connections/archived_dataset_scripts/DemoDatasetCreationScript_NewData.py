from data_connections.src.coco_tools import silt_to_coco, satsim_to_coco, merge_coco, silt_to_coco_panoptic, partition_dataset
from preprocess_functions import channel_mixture_A, channel_mixture_B, channel_mixture_C, adaptiveIQR, zscale, iqr_clipped, iqr_log, raw_file
from preprocess_functions import _median_column_subtraction, _median_row_subtraction, _background_subtract
from utilities import get_folders_in_directory, summarize_local_files, clear_local_caches, clear_local_cache, apply_bbox_corrections
import os
from utilities import clear_local_caches

final_data_path="/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets"


dirctoryA = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME01-2025Data"
final_outputA = os.path.join(final_data_path, f"RME01_2025")
dirctoryB = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04-2025Data"
final_outputB = os.path.join(final_data_path, f"RME04_2025")

all_origins = [dirctoryA, dirctoryB]
all_outputs = [final_outputA, final_outputB]

preprocess_func = iqr_log


for input_path, output_path in zip(all_origins, all_outputs):
    print(input_path)
    silt_to_coco(input_path, include_sats=True, include_stars=False, convert_png=True, process_func=preprocess_func, notes=f"Log_IQR_preprocessing for stability on new 2025 data from {input_path}")
    merge_coco([input_path], output_path, train_test_split=True, train_ratio=.80, val_ratio=.10, test_ratio=0.10, notes="80-10-10 train-val-test split")

    