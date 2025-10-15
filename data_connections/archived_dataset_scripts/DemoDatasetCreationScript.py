from coco_tools import silt_to_coco, satsim_to_coco, merge_coco, silt_to_coco_panoptic, partition_dataset
from preprocess_functions import channel_mixture_A, channel_mixture_B, channel_mixture_C, adaptiveIQR, zscale, iqr_clipped, iqr_log, raw_file
from preprocess_functions import _median_column_subtraction, _median_row_subtraction, _background_subtract
from utilities import get_folders_in_directory, summarize_local_files, clear_local_caches, clear_local_cache, apply_bbox_corrections
import os
from utilities import clear_local_caches

LA1 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-07-08"
LA2 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-07-09"
LA3 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-07-10"
LA4 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-07-11"
LA5 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-07-12"
LA6 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-07-13"
LA7 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-07-29"

LA8 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-08-04"
LA9 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-08-20"
LA10 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-09-13"
LA11 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-09-25"
LA12 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-10-06"
LA13 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-10-15"
LA14 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-10-23"
LA15 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-10-30"
LA16 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-11-07"
LA17 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-11-15"
LA18 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-11-26"
LA19 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-12-06"
LA20 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-12-17"
LA21 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-12-20"
LA22 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2024-12-30"
LA23 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2025-01-07"
LA24 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2025-01-10"
LA25 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2025-01-23"
LA26 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2025-05-03"
LA27 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2025-05-10"
LA28 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2025-05-16"
LA29 = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw/LMNT01Sat-2025-05-25"

final_data_path="/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets"

all_origins = [LA1, LA2, LA3, LA4, LA5, LA6, LA7, LA8, LA9, LA10, LA11, LA12, LA13, LA14, LA15, LA16, LA17, LA18, LA19, LA20, LA21, LA22, LA23, LA24, LA25, LA26, LA27, LA28, LA29]
training_set_output_path_LMNT01 = os.path.join(final_data_path, f"DemoLMNT01_WithTest")

preprocess_func = iqr_log


for path in all_origins:
    print(path)
    silt_to_coco(path, include_sats=True, include_stars=False, convert_png=True, process_func=preprocess_func, notes="Log_IQR_preprocessing for stability")

merge_coco(all_origins, training_set_output_path_LMNT01, train_test_split=True, train_ratio=.9, val_ratio=.1, test_ratio=.1, notes="merging all data for large draining set for SENTINEL demo")
