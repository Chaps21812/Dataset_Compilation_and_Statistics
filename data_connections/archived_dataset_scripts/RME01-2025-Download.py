from data_download_tool import download_data, summarize_s3_structure
from annotation_viewer import plot_annotations, plot_annotation_subset
from pandas_statistics import file_path_loader
from plots import *
from aws_s3_viewer import S3Client
import os


aws_directory = "third-party-data/PDS-RME04/Satellite/Annotations/"
s3_client = S3Client(aws_directory)

# RME01_Dates = ["2025-05-09","2025-05-12","2025-05-14","2025-05-16","2025-05-19","2025-05-21","2025-05-23","2025-05-28","2025-05-30","2025-06-02","2025-06-04","2025-06-06","2025-06-09","2025-06-11","2025-06-13","2025-06-16","2025-09-09","2025-09-10","2025-09-11","2025-09-12","2025-09-16","2025-09-17","2025-09-18","2025-09-19","2025-09-22","2025-09-23","2025-09-24","2025-09-25","2025-09-26"]
# RME01_Dates = ["2025-06-04","2025-06-06","2025-06-09","2025-06-11","2025-06-13","2025-06-16","2025-09-09","2025-09-10","2025-09-11","2025-09-12","2025-09-16","2025-09-17","2025-09-18","2025-09-19","2025-09-22","2025-09-23","2025-09-24","2025-09-25","2025-09-26"]
RME01_Dates = ["2025-09-12","2025-09-16","2025-09-17","2025-09-18","2025-09-19","2025-09-22","2025-09-23","2025-09-24","2025-09-25","2025-09-26"]

download_directory = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/RME01-2025Data"
for date in RME01_Dates:
    print(f"Search: {date}")
    s3_client.download_annotation_dates(date, download_directory)
