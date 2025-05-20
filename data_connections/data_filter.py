from pandas_statistics import file_path_loader

#Enter in the parameters you wish to download
dataset_directory = "/mnt/c/Users/david.chaparro/Documents/Repos/Dataset-Statistics/data/RME03Star"

#Local file handling tool
local_files = file_path_loader(dataset_directory)
#Pandas dataframes for referenece
image_attributes = local_files.statistics_file.sample_attributes
annotation_attributes = local_files.statistics_file.annotation_attributes

# local_files.delete_files_from_annotation(annotation_attributes[annotation_attributes['measured_snr'] < 5.0])
local_files.delete_files_from_sample(image_attributes[image_attributes['num_objects'] == 0].sample(700))