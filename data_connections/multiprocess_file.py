import multiprocessing as mp
import os
import sys

from src.raw_datset import raw_dataset

from data_connections.src.local_multiprocessing import SimpleFileLogger


def preprocess_large_dataset(Telescope_directory):
    # Create logger object
    file_logger = SimpleFileLogger(
        os.path.join(
            "/data/Dataset_Compilation_and_Statistics/logs",
            f"sidereal_download-{os.path.basename(Telescope_directory)}.log",
        )
    )
    sys.stdout = file_logger
    sys.stderr = file_logger
    print(
        f"Downloading Sidereal frames {os.path.basename(Telescope_directory)}...(PID={os.getpid()})"
    )

    calsat_dir = raw_dataset(Telescope_directory)
    calsat_dir.reinitialize_raw_dataset()
    calsat_dir.complete_calsat_dataset()


if __name__ == "__main__":
    calsat_main_folder = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/CalsatTesting_datasets"
    subcalsats = [
        os.path.join(calsat_main_folder, folder)
        for folder in os.listdir(calsat_main_folder)
    ]

    tasks = [
        mp.Process(target=preprocess_large_dataset, args=(dataset,))
        for dataset in subcalsats
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
