import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import os
import re
from typing import Union
from astropy.visualization import ZScaleInterval
import matplotlib.gridspec as gridspec
from preprocess_functions import _iqr_log

#This code was AI generated. I would love to spend time meticulously making plots, but I think my time is better spent analyzing them rather than adjusting details on a plot. 

def detect_column_type(series: pd.Series) -> str:
    """
    Detect whether a pandas Series (column) is categorical or numerical.

    Returns:
        "categorical" or "numerical"
    """
    # Drop NaNs for analysis
    s = series.dropna()

    if "time" in s.name.lower():
        return "time"
    if "file" in s.name.lower():
        return "file"
    if "path" in s.name.lower():
        return "file"

    # # If dtype is object or category, assume categorical
    # if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
    #     return "categorical"
    
    # If dtype is numeric, further check the number of unique values
    unique_vals = s.unique()
    num_unique = len(unique_vals)
    total = len(s)
    
    # Heuristic: if very few unique values relative to total, it's probably categorical
    if num_unique < 30:
        return "categorical"
    elif pd.api.types.is_numeric_dtype(s):
        return "numerical"
    # For boolean
    elif pd.api.types.is_bool_dtype(s):
        return "categorical"
    else:
        return "categorical"
                
def plot_categorical_column(series: pd.Series, filepath: str=None, dpi: int = 300 ):
    """
    Plot a bar chart for a categorical pandas Series with the mode, counts, and category name.
    """
    # Drop missing values
    s = series.dropna()

    # Count values
    value_counts = s.value_counts()
    mode_val = s.mode().iloc[0] if not s.mode().empty else "N/A"
    col_name = series.name if series.name else "Unnamed Column"

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(value_counts.index.astype(str), value_counts.values, color='teal', edgecolor='black')

    # Annotate bars with counts
    for bar, count in zip(bars, value_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(count),
                ha='center', va='bottom', fontsize=10)

    # Add title with mode and column name
    ax.set_title(f"Distribution of '{col_name}' (Mode: {mode_val})", fontsize=14)
    ax.set_ylabel("Count")
    ax.set_xlabel("Category")

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')

    # Create legend with counts
    legend_labels = [f"{cat}: {cnt}" for cat, cnt in value_counts.items()]
    ax.legend(legend_labels, title="Category Counts", loc='upper right')

    plt.tight_layout()
    if filepath is not None: 
        os.makedirs(filepath, exist_ok=True)
        
        # Sanitize the attribute name to create a safe filename
        safe_name = re.sub(r'[^\w\-_.]', '_', col_name.strip())
        filename = f"{safe_name}.png"
        full_path = os.path.join(filepath, filename)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        # print(f"Plot saved to: {full_path}")
    plt.show()
    
def plot_numerical_column(series: pd.Series, bins: int = 30, filepath: str=None, dpi: int = 300 ):
    """
    Plot a histogram for a numerical pandas Series with key statistics and column name.
    """
    # Drop missing values
    s = series.dropna()

    # Compute statistics
    mean_val = s.mean()
    median_val = s.median()
    std_val = s.std()
    min_val = s.min()
    max_val = s.max()
    col_name = series.name if series.name else "Unnamed Column"

    # Create histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(s, bins=bins, color='teal', edgecolor='black')

    # Add vertical lines for mean and median
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='blue', linestyle='dashed', linewidth=1.5, label=f'Median: {median_val:.2f}')

    # Set title and labels
    ax.set_title(f"Distribution of '{col_name}'", fontsize=14)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")

    # Create text box with stats
    stats_text = (f"Mean: {mean_val:.2f}\n"
                  f"Median: {median_val:.2f}\n"
                  f"Std Dev: {std_val:.2f}\n"
                  f"Min: {min_val:.2f}\n"
                  f"Max: {max_val:.2f}")
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    # Add legend for mean and median lines
    ax.legend(loc='upper left')

    plt.tight_layout()
    if filepath is not None: 
        os.makedirs(filepath, exist_ok=True)
        
        # Sanitize the attribute name to create a safe filename
        safe_name = re.sub(r'[^\w\-_.]', '_', col_name.strip())
        filename = f"{safe_name}.png"
        full_path = os.path.join(filepath, filename)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        # print(f"Plot saved to: {full_path}")
    plt.show()

def plot_scatter(x: pd.Series, y: pd.Series, alpha: float = 0.2, point_size: int = 40, color: str = 'teal', filepath: str=None, dpi: int = 300 ):
    """
    Create a scatter plot for two pandas Series with partial transparency to show point density.
    """
    # Align and drop missing values
    df = pd.concat([x, y], axis=1).dropna()
    x_vals = df.iloc[:, 0]
    y_vals = df.iloc[:, 1]
    x_name = x_vals.name if x_vals.name else "X"
    y_name = y_vals.name if y_vals.name else "Y"

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(x_vals, y_vals, alpha=alpha, s=point_size, color=color, edgecolor='black', linewidth=0.5)

    # Set labels and title
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"Scatter Plot of '{x_name}' vs '{y_name}'", fontsize=14)
    # Axis limits from 0 to 1
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add grid and layout
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if filepath is not None: 
        os.makedirs(filepath, exist_ok=True)
        
        # Sanitize the attribute name to create a safe filename
        safe_name = "Scatter_Plot"
        filename = f"{safe_name}.png"
        full_path = os.path.join(filepath, filename)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        # print(f"Plot saved to: {full_path}")
    plt.show()
    
def plot_lines(x1: pd.Series, y1: pd.Series, x2: pd.Series, y2: pd.Series,
               alpha: float = 0.2, color: str = 'teal', linewidth: float = 1.5, filepath: str=None, dpi: int = 300 ):
    """
    Plot line segments from (x1, y1) to (x2, y2) with transparency to visualize density.
    """
    # Combine into DataFrame and drop rows with missing values
    df = pd.concat([x1, y1, x2, y2], axis=1).dropna()
    col_names = df.columns.tolist()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for idx, row in df.iterrows():
        ax.plot([row[0], row[2]], [row[1], row[3]], color=color,
                alpha=alpha, linewidth=linewidth)

    # Axis labeling with fallback names
    x_label = f"{col_names[0]} → {col_names[2]}" if all(col_names) else "X"
    y_label = f"{col_names[1]} → {col_names[3]}" if all(col_names) else "Y"

    ax.set_title(f"Line Segments from ({col_names[0]}, {col_names[1]}) to ({col_names[2]}, {col_names[3]})", fontsize=14)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle='--', alpha=0.5)
    # Axis limits from 0 to 1
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    if filepath is not None: 
        os.makedirs(filepath, exist_ok=True)
        
        # Sanitize the attribute name to create a safe filename
        safe_name = "Line_Scatter_Plot"
        filename = f"{safe_name}.png"
        full_path = os.path.join(filepath, filename)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        # print(f"Plot saved to: {full_path}")
    plt.show()

def plot_time_column(series: pd.Series, bins: int = 30, filepath: str=None, dpi: int = 300 ):
    """
    Plot a histogram for a pandas Series of time data, showing key statistics such as mean, median, 
    std dev, min, and max.
    """
    # Ensure the series is datetime
    s = pd.to_datetime(series.dropna(), errors='coerce')
    
    # Compute statistics
    mean_val = s.mean()
    median_val = s.median()
    std_val = s.std()
    min_val = s.min()
    max_val = s.max()
    col_name = series.name if series.name else "Unnamed Column"

    # Create histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(s, bins=bins, color='teal', edgecolor='black')

    # Add vertical lines for mean and median
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val}')
    ax.axvline(median_val, color='blue', linestyle='dashed', linewidth=1.5, label=f'Median: {median_val}')

    # Set title and labels
    ax.set_title(f"Distribution of '{col_name}'", fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")

    # Format x-axis to show time properly
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))

    # Create text box with stats
    stats_text = (f"Mean: {mean_val}\n"
                  f"Median: {median_val}\n"
                  f"Std Dev: {std_val}\n"
                  f"Min: {min_val}\n"
                  f"Max: {max_val}")
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    # Add legend for mean and median lines
    ax.legend(loc='upper left')

    plt.tight_layout()
    if filepath is not None: 
        os.makedirs(filepath, exist_ok=True)
        
        # Sanitize the attribute name to create a safe filename
        safe_name = "times_Plot"
        filename = f"{safe_name}.png"
        full_path = os.path.join(filepath, filename)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        # print(f"Plot saved to: {full_path}")
    plt.show()

def plot_image_with_bbox(image: np.ndarray, x: int, y: int, size: int, object_n:int, full_file_path:str=None, dpi: int = 300, snr:tuple=None, alpha=.3, show:bool=False):
    """
    Plots an image with a square annotation and a padding of 100 pixels on all sides.

    Args:
        image (np.ndarray): The image to display.
        x (int): X-coordinate of the top-left corner of the annotation.
        y (int): Y-coordinate of the top-left corner of the annotation.
        size (int): Width/height of the square annotation.
    """
    # Image dimensions
    img_h, img_w = image.shape[:2]

    # Calculate padded annotation bounds
    pad = 100
    # Plot image
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray')

    # Draw the padded rectangle
    rect = patches.Rectangle(
        (x-size/2, y-size/2),
        size,
        size,
        linewidth=2,
        edgecolor='red',
        facecolor='none', 
        alpha=alpha
    )
    ax.add_patch(rect)
    ax.text(snr[0], snr[1], snr[2], fontsize=12, color='red', ha='right', va='bottom', alpha=alpha)
    plt.xlim(x-size-50,x+size+50)
    plt.ylim(y-size-50,y+size+50)

    # Turn off axes and show
    ax.set_axis_off()
    ax.set_title("Bounding Box Annotation")
    ax.set_axis_off()
    plt.tight_layout()
    if full_file_path is not None: 
        file = os.path.basename(full_file_path)
        direc = os.path.dirname(full_file_path)
        direc = os.path.dirname(direc)
        direc = os.path.join(direc, "annotation_view")
        os.makedirs(direc, exist_ok=True)
        
        # Sanitize the attribute name to create a safe filename
        safe_name = re.sub(r'[^\w\-_.]', '_', file.strip()).replace(".json","")
        filename = f"{safe_name}_{object_n}.png"
        full_path = os.path.join(direc, filename)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
    if show: plt.show()
    plt.close()
        # print(f"Plot saved to: {full_path}")

def plot_image_with_line(image: np.ndarray, x1: int, y1: int, x2: int, y2: int, object_n:int, full_file_path:str=None, dpi: int = 300, snr:tuple=None, alpha=.3, show:bool=False):
    """
    Plots an image with a line (x1, y1) -> (x2, y2) and a 100-pixel padded bounding box around the line.

    Args:
        image (np.ndarray): The image to display.
        x1, y1, x2, y2 (int): Coordinates of the line.
    """
    img_h, img_w = image.shape[:2]
    pad = 100

    # Get bounding box around the line
    min_x = max(0, min(x1, x2) - pad)
    max_x = min(img_w, max(x1, x2) + pad)
    min_y = max(0, min(y1, y2) - pad)
    max_y = min(img_h, max(y1, y2) + pad)

    # Compute width and height of padded box
    width = max_x - min_x
    height = max_y - min_y


    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray')
    ax.text(snr[0], snr[1], snr[2], fontsize=12, color='red', ha='right', va='bottom', alpha=alpha)
    plt.plot(snr[0], snr[1], 'ro', markersize=5, alpha=alpha)


    # Draw the line
    # ax.plot([x1, x2], [y1, y2], color='red', linewidth=2, label='Line', alpha=.2)
    plt.arrow(x1, y1, x2-x1, y2-y1, head_width=2, head_length=2, fc='red', ec='red', alpha=alpha)

    plt.xlim(min_x-width-50,max_x+width+50)
    plt.ylim(min_y-height-50,max_y+height+50)


    # Style
    ax.set_title("Line Segment Annotation")
    ax.set_axis_off()
    plt.tight_layout()
    if full_file_path is not None: 
        file = os.path.basename(full_file_path)
        direc = os.path.dirname(full_file_path)
        direc = os.path.dirname(direc)
        direc = os.path.join(direc, "annotation_view")
        os.makedirs(direc, exist_ok=True)
        
        # Sanitize the attribute name to create a safe filename
        safe_name = re.sub(r'[^\w\-_.]', '_', file.strip()).replace(".json","")
        filename = f"{safe_name}_{object_n}.png"
        full_path = os.path.join(direc, filename)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        # print(f"Plot saved to: {full_path}")
    if show: plt.show()
    plt.close()

def plot_image(image: np.ndarray, full_file_path:str=None, dpi: int = 300):
    """
    Plots an image with a line (x1, y1) -> (x2, y2) and a 100-pixel padded bounding box around the line.

    Args:
        image (np.ndarray): The image to display.
        x1, y1, x2, y2 (int): Coordinates of the line.
    """

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray')

    # Style
    ax.set_title("Base Image")
    ax.set_axis_off()
    ax.invert_yaxis()  # Invert y-axis to match the original image orientation
    plt.tight_layout()
    if full_file_path is not None: 
        file = os.path.basename(full_file_path)
        direc = os.path.dirname(full_file_path)
        direc = os.path.dirname(direc)
        direc = os.path.join(direc, "annotation_view")
        
        
        # Sanitize the attribute name to create a safe filename
        safe_name = re.sub(r'[^\w\-_.]', '_', file.strip()).replace(".json","")
        filename = f"{safe_name}_full.png"
        full_path = os.path.join(direc, filename)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        # print(f"Plot saved to: {full_path}")
    plt.close()

def plot_all_annotations(image: np.ndarray, annotations: list, img_size: tuple, full_file_path:str=None, dpi: int = 300):
    """
    Plots an image with all annotations.

    Args:
        image (np.ndarray): The image to display.
        annotations (list): List of annotation dictionaries.
    """
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray')

    # Draw all annotations
    for annotation in annotations:
        if annotation['class_name'] == "Satellite":
            x = annotation['x_center']*img_size[0]
            y = annotation['y_center']*img_size[1]
            size = annotation['x_max']*img_size[0] - annotation['x_min']*img_size[0]
            rect = patches.Rectangle(
                (x-size/2, y-size/2),
                size,
                size,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)
        elif annotation['class_name'] == "Star":
            if "x_start" in annotation.keys():
                x1 = annotation['x_start']*img_size[0]
                y1 = annotation['y_start']*img_size[1]
                x2 = annotation['x_end']*img_size[0]
                y2 = annotation['y_end']*img_size[1]
            else:
                x1 = annotation['x1']*img_size[0]
                y1 = annotation['y1']*img_size[1]
                x2 = annotation['x2']*img_size[0]
                y2 = annotation['y2']*img_size[1]
            plt.arrow(x1, y1, x2-x1, y2-y1, head_width=2, head_length=2, fc='red', ec='red', alpha=.2)
            # ax.plot([x1, x2], [y1, y2], color='red', linewidth=2, alpha=.2)

    # Style
    ax.set_title("Base Image with All Annotations")
    ax.set_axis_off()
    ax.invert_yaxis()  # Invert y-axis to match the original image orientation
    plt.tight_layout()
    if full_file_path is not None: 
        file = os.path.basename(full_file_path)
        direc = os.path.dirname(full_file_path)
        direc = os.path.dirname(direc)
        direc = os.path.join(direc, "annotation_view")
        os.makedirs(direc, exist_ok=True)
        
        # Sanitize the attribute name to create a safe filename
        safe_name = re.sub(r'[^\w\-_.]', '_', file.strip()).replace(".json","")
        filename = f"{safe_name}_all_annotations.png"
        full_path = os.path.join(direc, filename)
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(full_path, dpi=dpi, bbox_inches='tight')
        # print(f"Plot saved to: {full_path}")
    plt.close()

def plot_single_annotation(image: np.ndarray, bbox_old:tuple, bbox_new:tuple, title:str):
    """
    Plots an image with all annotations.

    Args:
        image (np.ndarray): The image to display.
        annotations (list): List of annotation dictionaries.
    """
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(z_scale_image(image), cmap='gray')

    # Draw all annotations
    rect = patches.Rectangle(
        (bbox_new[0], bbox_new[1]),
        bbox_new[2],
        bbox_new[3],
        linewidth=2,
        edgecolor='green',
        facecolor='none'
    )
    ax.add_patch(rect)
    ax.plot(bbox_new[0]+bbox_new[2]/2, bbox_new[1]+bbox_new[3]/2, '.', color="green", alpha=.5)
    # Draw all annotations
    rect = patches.Rectangle(
        (bbox_old[0], bbox_old[1]),
        bbox_old[2],
        bbox_old[3],
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)
    ax.plot(bbox_old[0]+bbox_old[2]/2, bbox_old[1]+bbox_old[3]/2, '.', color="red", alpha=.5)
    # Style
    ax.set_title(f"Adjusted Annotation: {title}")
    ax.set_axis_off()
    # ax.invert_yaxis()  # Invert y-axis to match the original image orientation
    plt.xlim(bbox_old[0]-50, bbox_old[0]+bbox_old[2]+50)
    plt.ylim(bbox_old[1]-50, bbox_old[1]+bbox_old[3]+50)
    plt.tight_layout()
    plt.show()

def plot_error_evaluator(image: np.ndarray, bboxes:tuple, index:int, attributes:dict):
    """
    Plots an image with all annotations.

    Args:
        image (np.ndarray): The image to display.
        annotations (list): List of annotation dictionaries.
    """

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])  # 2 rows, 2 cols

    ax_top = fig.add_subplot(gs[0, :])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_br = fig.add_subplot(gs[1, 1])

    # Plot
    fig.suptitle("Error Evaluation")

    # Plot the first image
    ax_top.imshow(_iqr_log(image), cmap='gray')
    ax_top.set_title(f'Image {attributes["fits_file"]}')

    # Plot the second image
    ax_bl.imshow(_iqr_log(image), cmap='gray')
    ax_bl.set_title(f'Annotation')

    # Add text in the third subplot
    ax_br.axis('off')
    text = ""
    for key,value in attributes.items():
        text = text+f"{key}: {value}\n"
    ax_br.text(0.01, 0.5, text, va='center', fontsize=12)

    if bboxes:
        spotlite_box = bboxes[index]
        rect = patches.Rectangle(
            (spotlite_box[0], spotlite_box[1]),
            spotlite_box[2],
            spotlite_box[3],
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax_bl.add_patch(rect)
        ax_bl.plot(spotlite_box[0]+spotlite_box[2]/2, spotlite_box[1]+spotlite_box[3]/2, '.', color="red", alpha=.5)
        ax_bl.set_xlim(spotlite_box[0]-20, spotlite_box[0]+spotlite_box[2]+20)
        ax_bl.set_ylim(spotlite_box[1]-20, spotlite_box[1]+spotlite_box[3]+20)

        for bbox in bboxes:
            # Draw all annotations
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax_top.add_patch(rect)
            ax_top.plot(bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2, '.', color="red", alpha=.5)
    plt.tight_layout()
    plt.show()

def z_scale_image(image:np.ndarray) -> np.ndarray:
    norm = ZScaleInterval(contrast=0.2)
    zscaled_data = norm(image)
    return zscaled_data