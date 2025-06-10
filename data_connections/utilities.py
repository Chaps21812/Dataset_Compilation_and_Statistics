from pandas_statistics import file_path_loader
import os

def get_folders_in_directory(directory_path):
  if not os.path.exists(directory_path):
    return []
  folders = [
      os.path.join(directory_path,entry.name) for entry in os.scandir(directory_path) if entry.is_dir()
  ]
  return folders

def view_folders(storage_path):
    for file in get_folders_in_directory(storage_path):
        #Local file handling tool
        local_files = file_path_loader(file)
        print(f"Num Samples: {len(local_files)}")

def clear_cache(storage_path):
    for file in get_folders_in_directory(storage_path):
        #Local file handling tool
        local_files = file_path_loader(file)
        local_files.clear_cache()

def recalculate_statistics(storage_path):
    for file in get_folders_in_directory(storage_path):
        #Local file handling tool
        local_files = file_path_loader(file)
        local_files.recalculate_statistics()
