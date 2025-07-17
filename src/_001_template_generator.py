# template_generator.py

import os
import re
import cv2
import json
import numpy as np

def generate_template_map(
    template_image_path: str,
    label_file_path: str,
    classes_file_path: str,
    anchor_name: str = "anchor_1",
    temp_dir: str = "./OMR_DEV/temp"
) -> str:
    """
    Generates a JSON file containing the relative positions of all objects
    in a template image with respect to a specified anchor.

    Args:
        template_image_path (str): Absolute or relative path to the template image file.
        label_file_path (str): Absolute or relative path to the YOLO-like label file (.txt) for the template.
        classes_file_path (str): Absolute or relative path to the classes.txt file.
        anchor_name (str): The name of the anchor class to use as a reference point.
                           Defaults to "anchor_1".
        temp_dir (str): Directory where the generated JSON file will be saved.
                        Defaults to "./OMR_DEV/temp".

    Returns:
        str: The absolute path to the generated template JSON file.
    """

    # Create the temporary directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)

    # Extract the base name of the template for naming the output JSON file
    template_name = os.path.basename(template_image_path)
    template_name = os.path.splitext(template_name)[0]

    # Load the template image
    template_image = cv2.imread(template_image_path)
    if template_image is None:
        raise FileNotFoundError(f"Template image not found at {template_image_path}")

    # Load the class names from the classes.txt file
    with open(classes_file_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    print(f"[INFO] Loaded {len(classes)} classes from {classes_file_path}.")

    # Get image dimensions for converting normalized YOLO coordinates to absolute pixels
    image_height, image_width = template_image.shape[:2]
    object_centers = {} # Stores (cx, cy) for each object
    object_boxes = {}   # Stores (x1, y1, x2, y2) for each object's bounding box

    # Parse the YOLO label file
    with open(label_file_path, 'r') as f:
        labels = f.readlines()

    for label in labels:
        parts = label.strip().split()
        class_id = int(parts[0])
        class_name = classes[class_id]

        # YOLO format provides normalized center_x, center_y, width, height
        cx_norm, cy_norm, w_norm, h_norm = map(float, parts[1:])

        # Convert normalized coordinates to absolute pixel coordinates
        abs_cx = int(cx_norm * image_width)
        abs_cy = int(cy_norm * image_height)
        abs_w = int(w_norm * image_width)
        abs_h = int(h_norm * image_height)

        # Calculate top-left (x1, y1) and bottom-right (x2, y2) corners of the bounding box
        x1 = int(abs_cx - abs_w / 2)
        y1 = int(abs_cy - abs_h / 2)
        x2 = int(abs_cx + abs_w / 2)
        y2 = int(abs_cy + abs_h / 2)

        object_centers[class_name] = (abs_cx, abs_cy)
        object_boxes[class_name] = (x1, y1, x2, y2)

    print(f"[INFO] Parsed {len(object_centers)} objects from {label_file_path}.")

    # --- Identify the anchor point ---
    if anchor_name not in object_centers:
        raise ValueError(f"Anchor '{anchor_name}' not found in the provided labels. Please ensure your classes.txt and label file define it.")
    anchor_center = object_centers[anchor_name]
    print(f"[INFO] Anchor '{anchor_name}' identified at original template coordinates: {anchor_center}")

    # --- Build the relative data structure ---
    # This dictionary will hold all the relative coordinates
    json_data = {
        "questions": {},
        "reg_no": {},
        "roll_no": {},
        "booklet_no": {}
    }

    # Helper function to parse class names and categorize them
    def parse_class_name_for_template(name):
        # Questions (e.g., "question_1")
        if re.match(r'^question_\d+$', name):
            return "question", int(name.split('_')[1])
        # Options (e.g., "1A", "5B")
        elif re.match(r'^\d+[A-D]$', name): # Assuming A, B, C, D options
            return "option", int(re.match(r'^(\d+)', name).group(1))
        # Registration number characters (e.g., "reg_no_0_5")
        elif name.startswith("reg_no_") and name != "reg_no": # Exclude the main "reg_no" group itself
            return "reg_no_char", name
        # Main registration number bounding box (if defined, e.g., "reg_no")
        elif name == "reg_no":
            return "reg_no_main", name
        # Roll number characters (e.g., "roll_no_0_3")
        elif name.startswith("roll_no_") and name != "roll_no":
            return "roll_no_char", name
        # Main roll number bounding box
        elif name == "roll_no":
            return "roll_no_main", name
        # Booklet number characters (e.g., "booklet_no_1_9")
        elif name.startswith("booklet_no_") and name != "booklet_no":
            return "booklet_no_char", name
        # Main booklet number bounding box
        elif name == "booklet_no":
            return "booklet_no_main", name
        else:
            return None, None # For classes not relevant to OMR processing

    # Iterate through all detected objects and calculate their relative positions
    for name in object_centers:
        if name == anchor_name: # Skip the anchor itself
            continue

        kind, identifier = parse_class_name_for_template(name)
        if kind is None: # Skip unrecognized classes
            continue

        cx, cy = object_centers[name]
        x1, y1, x2, y2 = object_boxes[name]

        # Calculate relative coordinates (dx, dy) and (x1_rel, y1_rel, x2_rel, y2_rel)
        rel_coords = {
            "center": {
                "dx": cx - anchor_center[0],
                "dy": cy - anchor_center[1]
            },
            "bbox": {
                "x1": x1 - anchor_center[0],
                "y1": y1 - anchor_center[1],
                "x2": x2 - anchor_center[0],
                "y2": y2 - anchor_center[1]
            }
        }

        # Populate the json_data dictionary based on the object's kind
        if kind == "question":
            qnum = identifier
            if qnum not in json_data["questions"]:
                json_data["questions"][qnum] = { "question": {}, "options": {} }
            json_data["questions"][qnum]["question"] = rel_coords
        elif kind == "option":
            qnum = identifier
            if qnum not in json_data["questions"]:
                json_data["questions"][qnum] = { "question": {}, "options": {} }
            json_data["questions"][qnum]["options"][name] = rel_coords
        elif kind in ["reg_no_char", "reg_no_main"]:
            json_data["reg_no"][identifier] = rel_coords
        elif kind in ["roll_no_char", "roll_no_main"]:
            json_data["roll_no"][identifier] = rel_coords
        elif kind in ["booklet_no_char", "booklet_no_main"]:
            json_data["booklet_no"][identifier] = rel_coords

    print("[INFO] Relative data structure for template elements built successfully.")

    # --- Save the relative structure to a JSON file ---
    output_json_path = os.path.join(temp_dir, f"{template_name}_template.json")
    with open(output_json_path, "w") as f:
        json.dump(json_data, f, indent=2) # indent for human-readable output

    print(f"[INFO] Template map saved to: {output_json_path}")
    return output_json_path

if __name__ == '__main__':
    # --- Example Usage (for testing template_generator.py directly) ---
    print("--- Running template_generator.py as main script ---")

    # Define paths for dummy files (replace with your actual paths for real use)
    _temp_dir = "temp_omr"
    _template_img_path = os.path.join(_temp_dir, "template.jpg")
    _label_file_path = os.path.join(_temp_dir, "template_labels.txt")
    _classes_file_path = os.path.join(_temp_dir, "classes.txt")

    # Generate the template map
    generated_template_map_path = generate_template_map(
        template_image_path=_template_img_path,
        label_file_path=_label_file_path,
        classes_file_path=_classes_file_path,
        anchor_name="anchor_1",
        temp_dir=_temp_dir
    )
    print(f"\n[SUCCESS] Template map generated at: {generated_template_map_path}")