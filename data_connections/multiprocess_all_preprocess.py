import multiprocessing as mp
import os
import sys

from src.coco_tools import COCODataset
from src.preprocess_functions import iqr_log_16bit, raw_file_16bit
from src.raw_datset import raw_dataset

from data_connections.src.local_multiprocessing import SimpleFileLogger


def preprocess_large_dataset(telescope_directory):
    # Create logger object
    file_logger = SimpleFileLogger(
        os.path.join(
            "/data/Dataset_Compilation_and_Statistics/logs",
            f"reprocess-{os.path.basename(telescope_directory)}.log",
        )
    )
    sys.stdout = file_logger
    sys.stderr = file_logger
    print(
        f"PRocessing Dataset {os.path.basename(telescope_directory)}...(PID={os.getpid()})"
    )

    telescope_basename = os.path.basename(telescope_directory)
    for date_folder in os.listdir(telescope_directory):
        subfolder = os.path.join(telescope_directory, date_folder)
        if os.path.isdir(subfolder):
            dataset = raw_dataset(subfolder)
            dataset.reinitialize_raw_dataset()
            dataset.recalculate_statistics()
            dataset.correct_annotations()
            coco_dataset = COCODataset(subfolder)
            coco_dataset.clear_extraneous_cache()
            coco_dataset.build_annotations()
            coco_dataset.generate_TTV_split(
                train_ratio=0.89, val_ratio=0.11, test_ratio=0
            )
            coco_dataset.move_fits_to_train_test_split(iqr_log_16bit)
    print("Complete, have a nice day, say an affirmation, eat some cheese.")


def remake_TTV_split_and_preprocess(telescope_directory):
    # Create logger object
    file_logger = SimpleFileLogger(
        os.path.join(
            "/data/Dataset_Compilation_and_Statistics/logs",
            f"preprocess_raw16bit-{os.path.basename(telescope_directory)}.log",
        )
    )
    sys.stdout = file_logger
    sys.stderr = file_logger
    print(
        f"PRocessing Dataset {os.path.basename(telescope_directory)}...(PID={os.getpid()})"
    )

    telescope_basename = os.path.basename(telescope_directory)
    for date_folder in os.listdir(telescope_directory):
        subfolder = os.path.join(telescope_directory, date_folder)
        if os.path.isdir(subfolder):
            # dataset = raw_dataset(subfolder)
            # dataset.reinitialize_raw_dataset()
            coco_dataset = COCODataset(subfolder)
            coco_dataset.archive_ttv_splits("log_IQR-16bit-89-11-0_TVT_split")
            coco_dataset.generate_TTV_split(
                train_ratio=0.80, val_ratio=0.10, test_ratio=0.10
            )
            coco_dataset.move_fits_to_train_test_split(raw_file_16bit)
    print("Complete, have a nice day, say an affirmation, eat some cheese.")


def preprocess_correct_large_dataset(telescope_directory):
    # Create logger object
    file_logger = SimpleFileLogger(
        os.path.join(
            "/data/Dataset_Compilation_and_Statistics/logs",
            f"sidereal_download-{os.path.basename(telescope_directory)}.log",
        )
    )
    sys.stdout = file_logger
    sys.stderr = file_logger
    print(
        f"PRocessing Dataset {os.path.basename(telescope_directory)}...(PID={os.getpid()})"
    )

    telescope_basename = os.path.basename(telescope_directory)
    for date_folder in os.listdir(telescope_directory):
        subfolder = os.path.join(telescope_directory, date_folder)
        if os.path.isdir(subfolder):
            dataset = raw_dataset(subfolder)
            dataset.reinitialize_raw_dataset()
            coco_dataset = COCODataset(subfolder)
            coco_dataset.build_annotations()
            coco_dataset.generate_TTV_split(
                train_ratio=0.80, val_ratio=0.10, test_ratio=0.10
            )
            coco_dataset.move_fits_to_train_test_split(iqr_log_16bit)
    print("Complete, have a nice day, say an affirmation, eat some cheese.")


if __name__ == "__main__":
    TELESCOPE_A = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME04-2025-Annotations"
    TELESCOPE_B = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME01-2025-Annotations"
    TELESCOPE_C = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT02-2025-Annotations"
    TELESCOPE_D = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01-2025-Annotations"
    TELESCOPE_E = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/ABQ01-2025-Annotations"

    ALL_TELESCOPES = [TELESCOPE_A, TELESCOPE_B, TELESCOPE_C, TELESCOPE_D, TELESCOPE_E]
    # ALL_TELESCOPES = [TELESCOPE_B]

    tasks = [
        mp.Process(target=remake_TTV_split_and_preprocess, args=(dataset,))
        for dataset in ALL_TELESCOPES
    ]

    try:
        print("Starting tasks")
        for p in tasks:
            p.start()

        for p in tasks:
            p.join()

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected. Terminating all tasks...")
        for p in tasks:
            p.terminate()
        for p in tasks:
            p.join()
        print("All tasks terminated.")
