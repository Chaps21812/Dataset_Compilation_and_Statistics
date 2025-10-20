from src.raw_datset import raw_dataset
from src.coco_tools import silt_to_coco, merge_coco
from src.preprocess_functions import raw_file
from src.s3client import S3Client
import os

if __name__ == "__main__":
    # # satsim_path = "/mnt/c/Users/david.chaparro/Documents/Repos/SatSim/output"
    # # local_satsim = satsim_path_loader(satsim_path)

    # dataset_directory = "/mnt/c/Users/david.chaparro/Documents/Repos/Dataset_Statistics/data/KWAJData"

    # #Local file handling tool
    # local_files = file_path_loader(dataset_directory)
    # local_files.recalculate_statistics()
    # print(f"Num Samples: {len(local_files)}")


    # path = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01_Raw"
    # for folder in os.listdir(path):
    #     local_files = raw_dataset(os.path.e(path,folder))
    #     local_files.create_calsat_dataset("/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/CalSatLMNT01-2024")

    # Calsat_path = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME01-2025Data"
    # calsats = raw_dataset(Calsat_path)
    # silt_to_coco(Calsat_path, process_func=raw_file)
    # merge_coco([Calsat_path], '/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/Finalized_datasets/merge_test', train_test_split=True, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1)

    ABQ01_PATH = "third-party-data/PDS-ABQ-01/Satellite/Annotations/"
    KWAJ01_PATH = "third-party-data/PDS-KWAJ01/Satellite/Annotations/"
    LMNT01_PATH= "third-party-data/PDS-LMNT01/Satellite/Annotations/"
    LMNT02_PATH = "third-party-data/PDS-LMNT02/Satellite/Annotations/"
    RME01_PATH ="third-party-data/PDS-RME01/Satellite/Annotations/"
    RME04_PATH ="third-party-data/PDS-RME04/Satellite/Annotations/"

    LMNT01_dates = ["2025-01-02","2025-01-03","2025-01-08","2025-01-09","2025-01-10","2025-01-13","2025-01-14","2025-01-15","2025-01-16","2025-01-17","2025-01-21","2025-01-22","2025-01-23","2025-01-24","2025-01-27","2025-01-28","2025-01-29","2025-01-30","2025-01-31","2025-02-03","2025-02-04","2025-02-05","2025-02-06","2025-02-07","2025-02-10","2025-02-11","2025-02-12","2025-02-13","2025-02-14","2025-02-18","2025-02-19","2025-02-20","2025-02-21","2025-02-24","2025-02-25","2025-02-26","2025-02-27","2025-02-28","2025-03-03","2025-03-04","2025-03-05","2025-03-06","2025-03-07","2025-03-10","2025-03-11","2025-03-12","2025-03-13","2025-03-14","2025-03-17","2025-03-18","2025-03-19","2025-03-20","2025-03-21","2025-03-24","2025-03-25","2025-03-26","2025-03-27","2025-03-28","2025-03-31","2025-04-01","2025-04-02","2025-04-03","2025-04-04","2025-04-07","2025-04-08","2025-04-09","2025-04-10","2025-04-11","2025-04-14","2025-04-15","2025-04-16","2025-04-17","2025-04-18","2025-04-21","2025-04-22","2025-04-23","2025-04-24","2025-04-25","2025-04-28","2025-04-29","2025-04-30","2025-05-01","2025-05-02","2025-05-05","2025-05-06","2025-05-07","2025-05-08","2025-05-13","2025-05-20","2025-05-27","2025-05-28","2025-06-03","2025-06-04","2025-06-10","2025-06-11"]
    downloader = S3Client(LMNT01_PATH)
    for date in LMNT01_dates:
        downloader.download_annotation_dates(date, "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/LMNT01-2025_Annotations")

    # downloader = S3Client(ABQ01_PATH)
    # downloader.download_annotation_dates("2025-10-20", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/abq_test")
    # downloader.download_annotation_dates("2025-10-17", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/abq_test")