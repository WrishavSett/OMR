import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

def load_template_and_coords(template_path, coords_path, class_path):
    """
    Loads the template image and its normalized anchor box coordinates.

    Args:
        template_path (str): The file path to the template image.
        coords_path (str): The file path to the annotation file with normalized coordinates.

    Returns:
        tuple: A tuple containing the template image (numpy.ndarray), its dimensions (height, width),
               and the list of coordinates.

    Raises:
        FileNotFoundError: If the template image or annotation file is not found.
    """
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template image not found at: {template_path}")
    if not os.path.exists(coords_path):
        raise FileNotFoundError(f"Annotation file not found at: {coords_path}")
    if not os.path.exists(class_path):
        raise FileNotFoundError(f"Class file not found at: {class_path}")

    template = cv2.imread(template_path)
    template_h, template_w, _ = template.shape

    with open(coords_path, 'r') as f:
        coords = [tuple(line.split()) for line in f]
    with open(class_path, 'r') as f:
        classes = [class_name.strip() for class_name in f]

    return template, template_h, template_w, coords, classes

def detect_anchor_boxes(image_path, template, template_w, template_h, coords, classes):
    """
    Detects anchor boxes in a given test image using template matching.

    Args:
        image_path (str): The file path to the test image.
        template (numpy.ndarray): The template image.
        template_w (int): The width of the template image.
        template_h (int): The height of the template image.
        coords (list): A list of normalized coordinates for the anchor boxes.

    Returns:
        tuple: A tuple containing the annotated image (numpy.ndarray) and a dictionary
               of detected anchor boxes. Returns (None, {}) if the image is not found.
    """
    if not os.path.exists(image_path):
        print(f"Error: Test image not found at {image_path}")
        return None, {}

    image = cv2.imread(image_path)
    detected_anchor_boxes = {}

    for i, coord in enumerate(coords):
        class_id, x_center_norm, y_center_norm, width_norm, height_norm = coord
        class_name = classes[int(class_id)]

        # Calculate pixel coordinates for the anchor box in the template
        x_center_temp = int(float(x_center_norm) * template_w)
        y_center_temp = int(float(y_center_norm) * template_h)
        width_temp = int(float(width_norm) * template_w)
        height_temp = int(float(height_norm) * template_h)

        x1_temp = int(x_center_temp - width_temp / 2)
        y1_temp = int(y_center_temp - height_temp / 2)
        x2_temp = int(x_center_temp + width_temp / 2)
        y2_temp = int(y_center_temp + height_temp / 2)

        # Extract the anchor box region from the template
        anchor_box_template = template[y1_temp:y2_temp, x1_temp:x2_temp]

        # Perform template matching on the loaded image
        result = cv2.matchTemplate(image, anchor_box_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # The top-left corner of the best match
        top_left = max_loc
        bottom_right = (top_left[0] + width_temp, top_left[1] + height_temp)

        # Calculate center point
        center_x = int((top_left[0] + bottom_right[0]) / 2)
        center_y = int((top_left[1] + bottom_right[1]) / 2)

        # Store the detected box coordinates and center point
        detected_anchor_boxes[class_name] = {
            "top_left": top_left,
            "bottom_right": bottom_right,
            "center": (center_x, center_y),
            "confidence": max_val
        }

        # Draw a rectangle around the detected anchor box on the image
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
        cv2.putText(image, f"{class_name}: {max_val:.2f}", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw center point
        cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)

        # Print the coordinates
        print(f"{class_name} Bounding Box: ({top_left[0]}, {top_left[1]}, {bottom_right[0]}, {bottom_right[1]})")
        print(f"{class_name} Center Point: ({center_x}, {center_y})")

    return image, detected_anchor_boxes

def display_image(image, window_name="Image", display_width=800):
    """
    Resizes and displays an image in a window.

    Args:
        image (numpy.ndarray): The image to display.
        window_name (str): The title of the display window.
        display_width (int): The desired width for the displayed image.
    """
    if image is None:
        print("Error: Cannot display empty image.")
        return

    h, w, _ = image.shape
    display_height = int(h * display_width / w)
    resized_image = cv2.resize(image, (display_width, display_height))

    cv2.imshow(window_name, resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- Main execution ---
if __name__ == "__main__":
    template_path = "./OMR/BLANK001.jpg"
    coords_path = "./OMR/BLANK001.txt"
    image_path = "./OMR/TEST/TEST-01010.jpg"
    class_path = "./OMR/classes.txt"  # Path to the class names file
    display_width = 480  # Set your desired display width here

    try:
        # 1. Load the template and coordinates using a function
        template, template_h, template_w, coords, classes = load_template_and_coords(template_path, coords_path, class_path)
        
        # 2. Detect anchor boxes and get the annotated image
        annotated_image, detected_boxes = detect_anchor_boxes(image_path, template, template_w, template_h, coords, classes)
        
        # 3. Display the result
        if annotated_image is not None:
            display_image(annotated_image, "Detected Anchor Boxes", display_width=display_width)

        # 4. Make the detected_boxes dictionary for further processing
        print("\nCoordinate Dictionary:")
        for name, data in detected_boxes.items():
            print(f"{name}: {data}")

    except FileNotFoundError as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")