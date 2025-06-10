from pandas_statistics import file_path_loader
from annotation_viewer import plot_annotations, plot_annotation_subset
from plots import *
import os
from IPython.display import clear_output

def get_folders_in_directory(directory_path) ->list:
  if not os.path.exists(directory_path):
    return []
  folders = [
      entry.name for entry in os.scandir(directory_path) if entry.is_dir()
  ]
  return folders

def delete_large_annotations(storage_path, area=1200):
    for file in get_folders_in_directory(storage_path):
        #Local file handling tool
        local_files = file_path_loader(os.path.join(storage_path,file))
        print(f"File: {file}, {len(local_files)}")

        image_attributes = local_files.statistics_file.sample_attributes
        annotation_attributes = local_files.statistics_file.annotation_attributes
        try: 
            large = len(annotation_attributes[annotation_attributes['area'] >= area])
            local_files.delete_files_from_sample(annotation_attributes[annotation_attributes['area'] >= area])
            print(f"Deleting: {large}/{len(annotation_attributes)} Large Annotation Images")
            regenerate_plots(os.path.join(storage_path,file))
        except KeyError: pass

def get_local_files(storage_path):
    total_images = 0
    for file in get_folders_in_directory(storage_path):
        #Local file handling tool
        local_files = file_path_loader(os.path.join(storage_path,file))
        total_images += len(local_files)
        print(f"Path: {file} Num Samples: {len(local_files)}")
    print(f"Total Samples: {total_images}")

def clear_local_caches(storage_path):
    for file in get_folders_in_directory(storage_path):
        #Local file handling tool
        local_files = file_path_loader(os.path.join(storage_path,file))
        local_files.clear_cache()

def clear_local_cache(storage_path):
    local_files = file_path_loader(storage_path)
    local_files.clear_cache()

def apply_bbox_corrections(storage_path):
    for file in get_folders_in_directory(storage_path):
        #Local file handling tool
        print(file)
        local_files = file_path_loader(os.path.join(storage_path,file))
        local_files.correct_annotations(apply_changes=True, require_approval=False)
        
def recalculate_all_statistics(storage_path):
    for file in get_folders_in_directory(storage_path):
        print(file)
        #Local file handling tool
        local_files = file_path_loader(os.path.join(storage_path,file))
        local_files.recalculate_statistics()

def generate_plots(storage_path):
    for file in get_folders_in_directory(storage_path):
        local_files = file_path_loader(os.path.join(storage_path,file))

        plots_save_path = os.path.join(storage_path,file, "plots")
        data_statistics = local_files.statistics_file
        #Plot all statistics collected in the file
        for col_name, col_data in data_statistics.sample_attributes.items():
            column_type = detect_column_type(col_data)
            print(column_type)
            if column_type == "categorical":
                try: 
                    plot_categorical_column(col_data, filepath=plots_save_path, dpi=500)
                    _clear_plots()
                except: print(f"Plotting Error {col_name}")
            elif column_type == "numerical":
                try: 
                    plot_numerical_column(col_data, filepath=plots_save_path, dpi=500)
                    _clear_plots()
                except: print(f"Plotting Error {col_name}")
            elif column_type == "time":
                try: 
                    plot_time_column(col_data, filepath=plots_save_path, dpi=500)
                    _clear_plots()
                except: print(f"Plotting Error {col_name}")
        for col_name, col_data in data_statistics.annotation_attributes.items():
            column_type = detect_column_type(col_data)
            if column_type == "categorical":
                try: 
                    plot_categorical_column(col_data, filepath=plots_save_path, dpi=500)
                    _clear_plots()
                except: print(f"Plotting Error {col_name}")
            elif column_type == "numerical":
                try: 
                    plot_numerical_column(col_data, filepath=plots_save_path, dpi=500)
                    _clear_plots()
                except: print(f"Plotting Error {col_name}")

        try:
            #Plot the x and y locations of the annotations
            x_locations=data_statistics.annotation_attributes["x_center"]
            y_locations=data_statistics.annotation_attributes["y_center"]
            plot_scatter(x_locations, y_locations, alpha=.05, filepath=plots_save_path, dpi=500)
            _clear_plots()
        except: print("Plotting Error: Centroid positions")

        #Plot line segments
        try: 
            plot_lines(data_statistics.annotation_attributes["x1"], data_statistics.annotation_attributes["y1"],
                    data_statistics.annotation_attributes["x2"], data_statistics.annotation_attributes["y2"],
                    filepath=plots_save_path, dpi=500, alpha=.10)
            _clear_plots()
        except: print("Plotting Error: Line Streaks")

def regenerate_plots(storage_path):
    local_files = file_path_loader(os.path.join(storage_path))

    plots_save_path = os.path.join(storage_path, "plots")
    data_statistics = local_files.statistics_file
    #Plot all statistics collected in the file
    for col_name, col_data in data_statistics.sample_attributes.items():
        column_type = detect_column_type(col_data)
        print(column_type)
        if column_type == "categorical":
            try: 
                plot_categorical_column(col_data, filepath=plots_save_path, dpi=500)
                _clear_plots()
            except: print(f"Plotting Error {col_name}")
        elif column_type == "numerical":
            try: 
                plot_numerical_column(col_data, filepath=plots_save_path, dpi=500)
                _clear_plots()
            except: print(f"Plotting Error {col_name}")
        elif column_type == "time":
            try: 
                plot_time_column(col_data, filepath=plots_save_path, dpi=500)
                _clear_plots()
            except: print(f"Plotting Error {col_name}")
    for col_name, col_data in data_statistics.annotation_attributes.items():
        column_type = detect_column_type(col_data)
        if column_type == "categorical":
            try: 
                plot_categorical_column(col_data, filepath=plots_save_path, dpi=500)
                _clear_plots()
            except: print(f"Plotting Error {col_name}")
        elif column_type == "numerical":
            try: 
                plot_numerical_column(col_data, filepath=plots_save_path, dpi=500)
                _clear_plots()
            except: print(f"Plotting Error {col_name}")

    try:
        #Plot the x and y locations of the annotations
        x_locations=data_statistics.annotation_attributes["x_center"]
        y_locations=data_statistics.annotation_attributes["y_center"]
        plot_scatter(x_locations, y_locations, alpha=.05, filepath=plots_save_path, dpi=500)
        _clear_plots()
    except: print("Plotting Error: Centroid positions")

    #Plot line segments
    try: 
        plot_lines(data_statistics.annotation_attributes["x1"], data_statistics.annotation_attributes["y1"],
                data_statistics.annotation_attributes["x2"], data_statistics.annotation_attributes["y2"],
                filepath=plots_save_path, dpi=500, alpha=.10)
        _clear_plots()
    except: print("Plotting Error: Line Streaks")

def _clear_plots():
    plt.clf()   # Clears the current figure
    plt.cla()   # Clears the current axes (optional)
    plt.close()
    clear_output(wait=True)