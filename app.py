#!/usr/bin/env python3

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import torch
import cv2
import csv
import os
import yaml
from pathlib import Path
from PIL import Image

from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.ndimage import binary_fill_holes, label
from scipy.ndimage import sum as ndimage_sum
from segment_anything import sam_model_registry, SamPredictor


MARKER_COLORS = {
    "Positive": (0, 100, 0),
    "Negative": (255, 28, 22)
}

MODELS = {
    "sam_vit_h_4b8939.pth": "vit_h",
    "sam_vit_l_0b3195.pth": "vit_l",
    "sam_vit_b_01ec64.pth": "vit_b"
}

def parse_config(file_path):
    try:
        # Resolve the absolute path of the config file
        config_path = Path(file_path).resolve()
        if not config_path.is_file():
            print(f"Error: Config file not found at '{config_path}'.")
            sys.exit()
        
        # Load the config file
        with config_path.open("r") as file:
            data = yaml.safe_load(file)

        # Check if config file is not empty
        if data is None:
            print(f"Error: Config file is empty.")
            sys.exit()
            
        # Validate required keys
        required_keys = ["checkpoint", "output-directory", "wing-cells"]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            print(f"Error: Missing required keys in config: {', '.join(missing_keys)}")
            sys.exit()

        # Resolve the absolute path for the output directory
        output_path = Path(data["output-directory"]).resolve()
        output_dir = str(output_path)
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
        else:
            print(f"Warning: Output directory already exists. Files might get overwritten.")
            
        # Resolve the absolute path for the checkpoint and identify the checkpoint
        checkpoint_path = Path(data["checkpoint"]).resolve()
        if not checkpoint_path.is_file():
            print(f"Error: Checkpoint file not found at '{checkpoint_path}'.")
            sys.exit()
        checkpoint_name = os.path.basename(checkpoint_path)
        if checkpoint_name in MODELS.keys():
            model_type = MODELS[checkpoint_name]
        else:
            print(f"Error: The checkpoint is not supported. Supported checkpoints: {', '.join(MODELS.keys())}")

        # Load wing cell data
        wing_cells = data["wing-cells"]
        sns_colors = sns.color_palette("hls", len(wing_cells))
        # Transform dictionary
        wing_segment_format = {}
        for i, (cell_id, display_name) in enumerate(wing_cells.items()):
            wing_segment_format[cell_id] = {
                "display_name": display_name,
                "color": sns_colors[i],
                "mask": None,
                "wing_area": None,
                "wing_height": None,
                "cell_area": None,
                "cell_perimeter": None
            }
    
        return output_dir, checkpoint_path, model_type, wing_segment_format

    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        sys.exit()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit()


def get_id_from_display_name(display_name, data):
    for key, value in data.items():
        if value.get("display_name") == display_name:
            return key
    return None


def display_image(image):
    return image


def reset_points(original_image):
    return original_image, original_image, []


def read_input_image(image_path):
    image_name = image_path.split("/")[-1]
    image = Image.open(image_path)
    image = np.asarray(image)
    return image, image, image, image_name


def sam_predict_mask(image, input_points, input_labels):
    predictor.set_image(image)

    masks, _, _ = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=False,
    )
    
    return masks[0]


def postprocess_mask(mask):
    labeled_mask, num_features = label(mask)
    if num_features == 0: 
        return mask
    component_sizes = ndimage_sum(mask, labeled_mask, range(1, num_features + 1))
    largest_component_label = np.argmax(component_sizes) + 1 
    largest_component_mask = labeled_mask == largest_component_label
    clean_mask = binary_fill_holes(largest_component_mask)
    
    return clean_mask


def generate_mask_image(image, wing_segments):
    # Create an empty RGBA image
    combined_masks = np.zeros((image.shape[0], image.shape[1], 4))  # 4 channels for RGBA

    # Loop through each wing segment in the dictionary
    for segment_name, segment_data in wing_segments.items():
        mask = segment_data["mask"]
        color = segment_data["color"]
        
        if mask is not None:
            # Apply the color to the mask (broadcasting over RGB channels)
            for c in range(3):  # RGB channels
                combined_masks[:, :, c] += mask * color[c]
            
            # Add to the alpha channel (set to 1 where the mask is present)
            combined_masks[:, :, 3] += mask

    # Normalize alpha values to stay within the range [0, 1]
    combined_masks[:, :, 3] = np.clip(combined_masks[:, :, 3], 0, 1)

    # Clip the RGB values to ensure they're within [0, 1] range
    combined_masks[:, :, :3] = np.clip(combined_masks[:, :, :3], 0, 1)
 
    # Create a combined image with the same dimensions
    height, width = image.shape[:2]
    dpi = 100
    fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
    ax.imshow(image)
    ax.imshow(combined_masks, alpha=0.4)
    ax.axis("off")

    # Convert figure to a NumPy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    # Convert the buffer to a NumPy array (RGBA)
    image_rgba = np.asarray(buf)
    plt.close(fig)

    # Convert to RGB
    image_rgba = image_rgba[..., :3]
    return image_rgba


def generate_mask(original_image, points, wing_segments):
    input_points = []
    input_labels = []
    cell = ""
    for (px, py, sel_mode, cell_type) in points:
        cell = cell_type
        cords = (int(px), int(py))
        if sel_mode == "Positive":
            input_points.append(cords)
            input_labels.append(1)
        if sel_mode == "Negative":
            input_points.append(cords)
            input_labels.append(0)

    input_points = np.array(input_points)
    input_labels = np.array(input_labels)

    mask = sam_predict_mask(original_image, input_points, input_labels)
    mask = postprocess_mask(mask)
    wing_segments[cell]["mask"] = mask
    wing_segments[cell]["wing_area"] = np.sum(mask)

    mask_image = generate_mask_image(original_image, wing_segments)
    return mask_image, wing_segments


def calculate_cell_features(image, wing_segments):
    # Grayscale the image 
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to get a binary image
    _, wing_thresh = cv2.threshold(blurred_image, 250, 255, cv2.THRESH_BINARY)

    # Invert the binary image
    wing_inv_thresh = cv2.bitwise_not(wing_thresh)

    # Find contour
    all_wing_contours, _ = cv2.findContours(wing_inv_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour based on area
    wing_contour = max(all_wing_contours, key=cv2.contourArea)
    
    # Calculate wing area
    wing_area = cv2.contourArea(wing_contour)

    # Find bounding Box
    x, y, w, h = cv2.boundingRect(wing_contour)

    # Draw a line around the wing
    wing_contour_image = gray.copy()
    cv2.drawContours(wing_contour_image, all_wing_contours, -1, (0), 10)

    # Loop through each wing segment in the dictionary
    for segment_name, segment_data in wing_segments.items():
        mask = segment_data["mask"]

        if mask is not None:
            # Fill out wing area and height
            segment_data["wing_area"] = wing_area
            segment_data["wing_height"] = h

            # Calculate the area
            segment_data["cell_area"] = np.sum(mask)

            # Calculate the perimeter
            binary_mask = mask.astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cell_contour = max(contours, key=cv2.contourArea)
            cell_perimeter = int(cv2.arcLength(cell_contour, closed=True))
            segment_data["cell_perimeter"] = cell_perimeter


def save_segmentation(image, wing_segments, image_name):
    # Calculate cell features
    wing_segments = calculate_cell_features(image, wing_segments)

    # Create or open a CSV file and write the header if it doesn't exist
    output_file_path = os.path.join(output_dir, "WingAreas.csv")
    header_written = os.path.exists(output_file_path)

    # Save 
    with open(output_file_path, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Filename", "VisibleWingAreaInPixels", "WingHeightInPixels", "Cell", "CellAreaInPixels", "CellPerimeterInPixels"])
        if not header_written:
            writer.writeheader()
        for segment_name, segment_data in wing_segments.items():
            writer.writerow({
                "Filename": image_name,
                "VisibleWingAreaInPixels": segment_data["wing_area"],
                "WingHeightInPixels": segment_data["wing_height"],
                "Cell": segment_name,
                "CellAreaInPixels": segment_data["wing_area"],
                "CellPerimeterInPixels": segment_data["cell_perimeter"]
            })

    output_subdir = output_dir + "/Wings/"
    os.makedirs(output_subdir, exist_ok=True)
    image = Image.fromarray(image)
    image.save(output_subdir + image_name)


def add_point(image, selection_mode, cell, points, evt: gr.SelectData):
    if image is None:
        return None, points

    # Extract coordinates from evt.index
    x = evt.index[0]
    y = evt.index[1]

    # Translate display name to cell id
    cell = get_id_from_display_name(cell, wing_segment_format)

    # Append new point
    new_points = points + [(x, y, selection_mode, cell)]

    image = image.copy()
    for (px, py, sel_mode, cell_type) in new_points:
        xi, yi = int(px), int(py)
        cv2.drawMarker(
            image, 
            (xi, yi), 
            MARKER_COLORS[sel_mode], 
            markerType=cv2.MARKER_TILTED_CROSS, 
            markerSize=10, 
            thickness=2
        )

    return image, image, new_points


if __name__ == "__main__":
    # Load the config file
    config_path = "config.yaml"  
    output_dir, checkpoint_path, model_type, wing_segment_format = parse_config(config_path)

    # Extract all "display_name" values
    display_names = [value["display_name"] for value in wing_segment_format.values()]

    # Ignore warnings
    warnings.filterwarnings("ignore")

    # Select the device for computation. Cuda is preferred if it is available.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Set up sam predictor checkpoint
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    
    predictor = SamPredictor(sam)

    # Create demo
    with gr.Blocks() as demo:
        points = gr.State([])
        original_image = gr.State(None)
        point_image = gr.State(None)
        image_name = gr.State(None)
        wing_segments = gr.State(wing_segment_format)

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload your image here", 
                    type="filepath", 
                    sources=["upload", "clipboard"],
                    height=700, 
                    interactive=True
                )
                
            with gr.Column():
                gr.Markdown("# Segment Anywing\n"
                            "[Segment Anything](https://segment-anything.com/) based segmentation tool by Jakob Materna")
                output_image = gr.Image(
                    label="Segmented Image", 
                    height=335,
                    interactive=False)
                with gr.Row():
                    cell_options = gr.Dropdown(
                        choices=display_names, 
                        label="Wing Cell"
                    )
                    selection_options = gr.Radio(
                        choices=["Positive", "Negative"],
                        value="Positive", 
                        label="Selection method"
                    )
                undo_button = gr.Button("Clear Selection")
                generate_mask_button = gr.Button("Generate Mask")
                save_segmentation_button = gr.Button("Save Segmentation")

        input_image.upload(
            fn=read_input_image,
            inputs=input_image,
            outputs=[output_image, original_image, point_image, image_name]
        )

        input_image.select(
            fn=add_point,
            inputs=[point_image, selection_options, cell_options, points],
            outputs=[input_image, point_image, points]
        )

        cell_options.input(
            fn=reset_points,
            inputs=[original_image],
            outputs=[input_image, point_image, points]
        )

        undo_button.click(
            fn=reset_points,
            inputs=[original_image],
            outputs=[input_image, point_image, points]
        )

        generate_mask_button.click(
            fn=generate_mask,
            inputs=[original_image, points, wing_segments],
            outputs=[output_image, wing_segments]
        )

        save_segmentation_button.click(
            fn=save_segmentation,
            inputs=[output_image, wing_segments, image_name]
        )

    demo.launch(server_name="127.0.0.1", debug=True)
