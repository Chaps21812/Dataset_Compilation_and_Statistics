from src.raw_datset import raw_dataset
from src.coco_tools import silt_to_coco, merge_coco
from src.preprocess_functions import raw_file
from src.s3client import S3Client
import os

if __name__ == "__main__":
    ABQ01_PATH = "third-party-data/PDS-ABQ-01/Satellite/Annotations/"
    KWAJ01_PATH = "third-party-data/PDS-KWAJ01/Satellite/Annotations/"
    LMNT01_PATH= "third-party-data/PDS-LMNT01/Satellite/Annotations/"
    LMNT02_PATH = "third-party-data/PDS-LMNT02/Satellite/Annotations/"
    RME01_PATH ="third-party-data/PDS-RME01/Satellite/Annotations/"
    RME04_PATH ="third-party-data/PDS-RME04/Satellite/Annotations/"


    RME01_dates= ["2025-05-09", "2025-05-12", "2025-05-14", "2025-05-16", "2025-05-19", "2025-05-21", "2025-05-23", "2025-05-28", "2025-05-30", "2025-06-02", "2025-06-04", "2025-06-06", "2025-06-09", "2025-06-11", "2025-06-13", "2025-06-16", "2025-09-09", "2025-09-10", "2025-09-11", "2025-09-12", "2025-09-16", "2025-09-17", "2025-09-18", "2025-09-19", "2025-09-22", "2025-09-23", "2025-09-24", "2025-09-25", "2025-09-26", "2025-09-29", "2025-09-30", "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-06", "2025-10-07", "2025-10-08", "2025-10-09", "2025-10-10", "2025-10-14", "2025-10-15", "2025-10-16", "2025-10-17", "2025-10-20"]
    # RME01_dates= ["2025-06-16", "2025-09-09", "2025-09-10", "2025-09-11", "2025-09-12", "2025-09-16", "2025-09-17", "2025-09-18", "2025-09-19", "2025-09-22", "2025-09-23", "2025-09-24", "2025-09-25", "2025-09-26", "2025-09-29", "2025-09-30", "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-06", "2025-10-07", "2025-10-08", "2025-10-09", "2025-10-10", "2025-10-14", "2025-10-15", "2025-10-16", "2025-10-17", "2025-10-20"]
    # RME01_dates= ["2025-06-16", "2025-09-09", "2025-09-10", "2025-09-11", "2025-09-12", "2025-09-16", "2025-09-17", "2025-09-18", "2025-09-19", "2025-09-22", "2025-09-23", "2025-09-24", "2025-09-25", "2025-09-26", "2025-09-29", "2025-09-30", "2025-10-01", "2025-10-02", "2025-10-03", "2025-10-06"]
    # RME01_dates= ["2025-06-16", "2025-09-09", "2025-09-10", "2025-09-11", "2025-09-12", "2025-09-16", "2025-09-17", "2025-09-18"]


    downloader = S3Client(RME01_PATH)
    for date in reversed(RME01_dates):
        print(date)
        # downloader.download_annotation_dates_Sidereal(date, "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME01-2025_Annotations")
        downloader.download_annotation_dates_calsats("2024-08-08", "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/SiderealTestDownload")

