# omr_processor.py

import os
import re
import cv2
import json
import numpy as np
import pandas as pd # Not strictly used in provided blocks but often useful for OMR data

# Import the template generation function
try:
    from _001_template_generator import generate_template_map
except ImportError:
    print("[ERROR] Could not import 'generate_template_map'. Please ensure 'template_generator.py' is in the same directory or your Python path.")
    print("Consider running 'template_generator.py' directly once to ensure the template map is created.")
    exit(1) # Exit if essential module cannot be imported

# --- Utility Function: Get mean intensity of a bounding box ---
def get_mean_intensity(image, bbox):
    """
    Calculates the mean pixel intensity within a given bounding box.
    Expects a grayscale image or converts a color ROI to grayscale.

    Args:
        image (np.array): The input image (can be BGR or grayscale).
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).

    Returns:
        float: The mean pixel intensity (0-255), or 0 if the bbox is invalid.
    """
    x1, y1, x2, y2 = map(int, bbox)
    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    if x2 <= x1 or y2 <= y1: # Check for invalid or empty bounding box
        return 0.0 # Return 0 for invalid regions

    roi = image[y1:y2, x1:x2]
    # If the ROI is a color image, convert it to grayscale before calculating mean
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return np.mean(roi)

# --- Function: Detect all marked bubbles with dynamic threshold ---
def detect_marked_bubble(image, bubble_group_data, anchor_x, anchor_y):
    """
    Detects all marked bubbles within a group using a dynamic threshold
    based on relative intensities. If there's only one bubble, it uses an
    absolute intensity threshold.

    Args:
        image (np.array): The test image.
        bubble_group_data (dict): A dictionary of bubble names and their relative bbox data.
        anchor_x (int): Absolute X-coordinate of the anchor in the test image.
        anchor_y (int): Absolute Y-coordinate of the anchor in the test image.

    Returns:
        list: A list of names of all detected marked bubbles.
    """
    intensities = {}
    relative_intensities = {}
    all_relative_intensity_values = []
    marked_bubbles = []

    # Calculate absolute bounding boxes and mean intensities for each bubble in the group
    for bubble_name, bubble_rel_data in bubble_group_data.items():
        bbox_rel = bubble_rel_data["bbox"]
        x1_abs = int(anchor_x + bbox_rel["x1"])
        y1_abs = int(anchor_y + bbox_rel["y1"])
        x2_abs = int(anchor_x + bbox_rel["x2"])
        y2_abs = int(anchor_y + bbox_rel["y2"])

        current_bbox = (x1_abs, y1_abs, x2_abs, y2_abs)
        mean_intensity = get_mean_intensity(image, current_bbox)
        intensities[bubble_name] = mean_intensity

    total_intensity = sum(intensities.values())

    # Determine relative intensities and collect them for dynamic threshold calculation
    if total_intensity > 0: # Avoid division by zero if all bubbles are completely dark
        for bubble_name, mean_intensity in intensities.items():
            relative_intensity = (mean_intensity / total_intensity) * 100
            relative_intensities[bubble_name] = relative_intensity
            all_relative_intensity_values.append(relative_intensity)

        # --- Dynamic Threshold Calculation ---
        if len(all_relative_intensity_values) > 1: # Apply dynamic threshold only if there are multiple elements
            min_val = min(all_relative_intensity_values)
            max_val = max(all_relative_intensity_values)
            dynamic_threshold = (((min_val + max_val) / 2) - 2) # Adjusted threshold

            # Identify marked bubbles based on dynamic threshold
            for bubble_name, relative_intensity in relative_intensities.items():
                if relative_intensity < dynamic_threshold: # Bubbles with lower intensity are considered marked
                    marked_bubbles.append(bubble_name)

        else: # This branch covers cases where len(all_relative_intensity_values) is 0 or 1
            if len(all_relative_intensity_values) == 1:
                # If there's only one element in the group, use a fixed absolute mean intensity threshold
                single_bubble_name = list(intensities.keys())[0] # Get the name of the single bubble
                single_bubble_mean_intensity = intensities[single_bubble_name] # Get its mean intensity

                if single_bubble_mean_intensity < 190: # Fixed threshold for single bubble
                    marked_bubbles.append(single_bubble_name)
            # If len is 0, no bubbles to process, marked_bubbles remains empty
    return marked_bubbles

# --- Generalized Visualization Function ---
def draw_label_group(image, label_group, anchor_x, anchor_y, color, label_prefix=""):
    """
    Draws circles (for centers), text labels, and rectangles (for bounding boxes)
    for a group of elements on the given image.

    Args:
        image (np.array): The image to draw on.
        label_group (dict): Dictionary of element names and their relative data from template.
        anchor_x (int): Absolute X-coordinate of the anchor.
        anchor_y (int): Absolute Y-coordinate of the anchor.
        color (tuple): BGR color (e.g., (0, 0, 255) for red) for drawing.
        label_prefix (str): Optional prefix for the text label (e.g., "REG_").
    """
    for name, data in label_group.items():
        # Calculate absolute center coordinates
        dx, dy = data["center"]["dx"], data["center"]["dy"]
        cx, cy = int(anchor_x + dx), int(anchor_y + dy)

        cv2.circle(image, (cx, cy), 4, color, -1) # Draw filled circle at center
        cv2.putText(image, f"{label_prefix}{name}", (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1) # Draw text label

        # Draw bounding box if present in data
        if "bbox" in data:
            x1 = int(anchor_x + data["bbox"]["x1"])
            y1 = int(anchor_y + data["bbox"]["y1"])
            x2 = int(anchor_x + data["bbox"]["x2"])
            y2 = int(anchor_y + data["bbox"]["y2"])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1) # Draw rectangle

def process_omr_sheet(
    test_image_path: str,
    template_image_path: str,
    label_file_path: str,
    classes_file_path: str,
    anchor_name: str = "anchor_1",
    temp_dir: str = "temp_omr"
) -> dict:
    """
    Processes an OMR test image using a template to detect marked answers
    and other data fields (Registration, Roll, Booklet Numbers).

    Args:
        test_image_path (str): Absolute or relative path to the test image file (scanned OMR sheet).
        template_image_path (str): Absolute or relative path to the template image file.
        label_file_path (str): Absolute or relative path to the YOLO-like label file (.txt) for the template.
        classes_file_path (str): Absolute or relative path to the classes.txt file.
        anchor_name (str): The name of the anchor class used as a reference point.
                           Defaults to "anchor_1".
        temp_dir (str): Directory to save intermediate files (template map, detected answers JSON).
                        Defaults to "temp_omr".

    Returns:
        dict: A dictionary containing all detected answers and data.
    """
    # Ensure temp directory exists
    os.makedirs(temp_dir, exist_ok=True)

    # --- Step 1: Generate or retrieve the template map path ---
    # This calls the function from template_generator.py to ensure the template map exists
    template_map_path = generate_template_map(
        template_image_path=template_image_path,
        label_file_path=label_file_path,
        classes_file_path=classes_file_path,
        anchor_name=anchor_name,
        temp_dir=temp_dir
    )
    print(f"\n[INFO] Using template map from: {template_map_path}")

    # --- Load Test Image ---
    test_name = os.path.basename(test_image_path)
    test_name = os.path.splitext(test_name)[0]
    print(f"[INFO] Processing test image: {test_name}")
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        raise FileNotFoundError(f"Test image not found at {test_image_path}")

    # Load template image (required for ORB feature extraction)
    template_image = cv2.imread(template_image_path)
    if template_image is None:
        raise FileNotFoundError(f"Template image not found at {template_image_path}")

    # --- Step 2: Find anchor in test image using ORB + Homography ---
    print("[INFO] Detecting anchor in test image using ORB features...")
    orb = cv2.ORB_create(5000) # Create ORB detector with 5000 features
    kp1, des1 = orb.detectAndCompute(template_image, None) # Keypoints and descriptors for template
    kp2, des2 = orb.detectAndCompute(test_image, None)     # Keypoints and descriptors for test image

    bf = cv2.BFMatcher(cv2.NORM_HAMMING) # Brute-Force Matcher with Hamming distance for ORB
    matches = bf.knnMatch(des1, des2, k=2) # Find 2 best matches for each descriptor

    # Apply ratio test to find good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance: # Ratio test: ensures distinctiveness of matches
            good_matches.append(m)

    print(f"[INFO] Found {len(good_matches)} good matches for anchor detection.")

    # Compute Homography if enough good matches are found
    if len(good_matches) > 10: # A minimum number of matches is required for stable homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # M is the 3x3 homography matrix
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) # RANSAC for robustness against outliers

        # Re-parse template labels to get the original anchor center in template image
        # This is necessary because the template map stores relative, not original absolute coords
        image_height_temp, image_width_temp = template_image.shape[:2]
        temp_object_centers = {}
        with open(classes_file_path, 'r') as f_classes:
            temp_classes = [line.strip() for line in f_classes.readlines()]
        with open(label_file_path, 'r') as f_labels:
            temp_labels = f_labels.readlines()

        original_anchor_center_in_template = None
        for label in temp_labels:
            parts = label.strip().split()
            class_id = int(parts[0])
            class_name = temp_classes[class_id]
            if class_name == anchor_name:
                cx_norm, cy_norm, _, _ = map(float, parts[1:])
                original_anchor_center_in_template = (int(cx_norm * image_width_temp), int(cy_norm * image_height_temp))
                break # Anchor found, exit loop

        if original_anchor_center_in_template is None:
             raise ValueError(f"Anchor '{anchor_name}' not found when re-parsing template labels for homography. Check labels and classes.")

        # Transform the anchor point from template coordinates to test image coordinates
        anchor_pt = np.array([[original_anchor_center_in_template]], dtype=np.float32)
        transformed_anchor = cv2.perspectiveTransform(anchor_pt, M)
        transformed_center = tuple(map(int, transformed_anchor[0][0])) # The new absolute anchor position
        print(f"[INFO] Anchor '{anchor_name}' successfully located in test image at: {transformed_center}")

        # Create a copy of the test image to draw results on
        result_image = test_image.copy()

        # Draw the detected anchor point on the result image for visualization
        cv2.circle(result_image, transformed_center, 5, (0, 0, 255), -1) # Red filled circle
        cv2.putText(result_image, anchor_name, (transformed_center[0] + 10, transformed_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


    else:
        raise Exception(f"[ERROR] Not enough good matches found ({len(good_matches)}) for homography to locate the anchor. Image alignment failed.")

    # --- Step 3: Load the relative template data ---
    with open(template_map_path, "r") as f:
        template_data = json.load(f)

    # Set the anchor's absolute coordinates for all subsequent calculations
    anchor_x, anchor_y = transformed_center
    print(f"[INFO] Loaded template data. All relative coordinates will be mapped using anchor at ({anchor_x}, {anchor_y}).")


    # --- Step 4: Process and detect marked bubbles for different groups ---
    detected_answers = {}

    # Questions and their options
    for qnum_str in sorted(template_data["questions"].keys(), key=int): # Iterate in numerical order
        qnum = int(qnum_str)
        qdata = template_data["questions"][qnum_str]
        if "options" in qdata and qdata["options"]:
            print(f"\n[Processing] Question {qnum} Options:")
            marked_options = detect_marked_bubble(result_image, qdata["options"], anchor_x, anchor_y)
            if marked_options:
                print(f"[Result] Question {qnum} Answer(s): {', '.join(marked_options)}")
                detected_answers[f"question_{qnum}"] = marked_options # Store as a list to allow multiple marks
            else:
                print(f"[Result] Question {qnum}: No clear marked option detected.")
                detected_answers[f"question_{qnum}"] = None # Set to None if no options are marked

    # Registration Number
    # Group individual digit bubbles by their column index (e.g., reg_no_0_X, reg_no_1_X)
    reg_no_chars_groups = {}
    for name, data in template_data["reg_no"].items():
        if name.startswith("reg_no_") and name != "reg_no": # Exclude the main "reg_no" label if it exists
            match = re.match(r'reg_no_(\d+)_(\d+)', name)
            if match:
                group_index = int(match.group(1)) # This is the column index
                if group_index not in reg_no_chars_groups:
                    reg_no_chars_groups[group_index] = {}
                reg_no_chars_groups[group_index][name] = data

    print("\n[Processing] Registration Number:")
    detected_reg_no_list = [] # Will store detected digits for each column
    for group_index in sorted(reg_no_chars_groups.keys()): # Process columns in order
        # print(f"  Processing reg_no_column_{group_index}:") # Detailed log, can be uncommented for debugging
        marked_chars = detect_marked_bubble(result_image, reg_no_chars_groups[group_index], anchor_x, anchor_y)

        column_digits = []
        if marked_chars:
            # Sort marked_chars to ensure stable output, especially if multiple bubbles are marked in a column
            for marked_char_name in sorted(marked_chars):
                # Extract the actual digit from the bubble name (e.g., '0' from 'reg_no_0_0')
                digit_match = re.search(r'\_(\d+)$', marked_char_name)
                if digit_match:
                    # print(f"    Detected digit: {digit_match.group(1)}") # Detailed log
                    column_digits.append(digit_match.group(1))
                else:
                    column_digits.append("?") # Placeholder if digit cannot be extracted from name
        
        if column_digits:
            detected_reg_no_list.append("".join(column_digits)) # Join multiple marked digits if error occurs in one column
        else:
            detected_reg_no_list.append("-") # Placeholder for an unmarked or undected column

    final_reg_no = "".join(detected_reg_no_list)
    if all(char == '-' for char in final_reg_no):
        detected_answers["reg_no"] = None # Set to None if all columns are unmarked
    else:
        detected_answers["reg_no"] = final_reg_no
    print(f"[Result] Detected Registration Number: {detected_answers['reg_no']}")


    # Roll Number (logic is identical to Registration Number)
    roll_no_chars_groups = {}
    for name, data in template_data["roll_no"].items():
        if name.startswith("roll_no_") and name != "roll_no":
            match = re.match(r'roll_no_(\d+)_(\d+)', name)
            if match:
                group_index = int(match.group(1))
                if group_index not in roll_no_chars_groups:
                    roll_no_chars_groups[group_index] = {}
                roll_no_chars_groups[group_index][name] = data

    print("\n[Processing] Roll Number:")
    detected_roll_no_list = []
    for group_index in sorted(roll_no_chars_groups.keys()):
        # print(f"  Processing roll_no_column_{group_index}:")
        marked_chars = detect_marked_bubble(result_image, roll_no_chars_groups[group_index], anchor_x, anchor_y)
        
        column_digits = []
        if marked_chars:
            for marked_char_name in sorted(marked_chars):
                digit_match = re.search(r'\_(\d+)$', marked_char_name)
                if digit_match:
                    column_digits.append(digit_match.group(1))
        
        if column_digits:
            detected_roll_no_list.append("".join(column_digits))
        else:
            detected_roll_no_list.append("-")

    final_roll_no = "".join(detected_roll_no_list)
    if all(char == '-' for char in final_roll_no):
        detected_answers["roll_no"] = None
    else:
        detected_answers["roll_no"] = final_roll_no
    print(f"[Result] Detected Roll Number: {detected_answers['roll_no']}")

    # Booklet Number (logic is identical to Registration Number)
    booklet_no_chars_groups = {}
    for name, data in template_data["booklet_no"].items():
        if name.startswith("booklet_no_") and name != "booklet_no":
            match = re.match(r'booklet_no_(\d+)_(\d+)', name)
            if match:
                group_index = int(match.group(1))
                if group_index not in booklet_no_chars_groups:
                    booklet_no_chars_groups[group_index] = {}
                booklet_no_chars_groups[group_index][name] = data

    print("\n[Processing] Booklet Number:")
    detected_booklet_no_list = []
    for group_index in sorted(booklet_no_chars_groups.keys()):
        # print(f"  Processing booklet_no_column_{group_index}:")
        marked_chars = detect_marked_bubble(result_image, booklet_no_chars_groups[group_index], anchor_x, anchor_y)
        
        column_digits = []
        if marked_chars:
            for marked_char_name in sorted(marked_chars):
                digit_match = re.search(r'\_(\d+)$', marked_char_name)
                if digit_match:
                    column_digits.append(digit_match.group(1))
        
        if column_digits:
            detected_booklet_no_list.append("".join(column_digits))
        else:
            detected_booklet_no_list.append("-")

    final_booklet_no = "".join(detected_booklet_no_list)
    if all(char == '-' for char in final_booklet_no):
        detected_answers["booklet_no"] = None
    else:
        detected_answers["booklet_no"] = final_booklet_no
    print(f"[Result] Detected Booklet Number: {detected_answers['booklet_no']}")

    print("\n--- Detected Answers Summary ---")
    for key, value in detected_answers.items():
        print(f"{key}: {value}")


    # --- Step 5: Draw centers and bounding boxes on the result image for visualization ---
    print("\n[INFO] Drawing detected regions and labels on the result image...")
    # Questions
    for qnum_str, qdata in template_data["questions"].items():
        qnum = int(qnum_str) # Convert to int for sorting/indexing

        # Draw main question bounding box (if defined in template)
        if "question" in qdata:
            qrel = qdata["question"]
            dx, dy = qrel["center"]["dx"], qrel["center"]["dy"]
            cx, cy = int(anchor_x + dx), int(anchor_y + dy)
            cv2.circle(result_image, (cx, cy), 4, (255, 0, 255), -1) # Magenta center
            cv2.putText(result_image, f"Q{qnum}", (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            if "bbox" in qrel:
                x1 = int(anchor_x + qrel["bbox"]["x1"])
                y1 = int(anchor_y + qrel["bbox"]["y1"])
                x2 = int(anchor_x + qrel["bbox"]["x2"])
                y2 = int(anchor_y + qrel["bbox"]["y2"])
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 255), 2) # Magenta rectangle

        # Draw options for the question
        for opt_name, opt_data in qdata.get("options", {}).items():
            dx, dy = opt_data["center"]["dx"], opt_data["center"]["dy"]
            ox, oy = int(anchor_x + dx), int(anchor_y + dy)
            cv2.circle(result_image, (ox, oy), 4, (0, 255, 255), -1) # Yellow center
            cv2.putText(result_image, opt_name, (ox + 5, oy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if "bbox" in opt_data:
                x1 = int(anchor_x + opt_data["bbox"]["x1"])
                y1 = int(anchor_y + opt_data["bbox"]["y1"])
                x2 = int(anchor_x + opt_data["bbox"]["x2"])
                y2 = int(anchor_y + opt_data["bbox"]["y2"])
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 255), 2) # Yellow rectangle

    # Draw Registration Number bubbles
    draw_label_group(result_image, template_data["reg_no"], anchor_x, anchor_y, (0, 100, 255), "REG_") # Orange color
    # Draw Roll Number bubbles
    draw_label_group(result_image, template_data["roll_no"], anchor_x, anchor_y, (100, 255, 0), "ROLL_") # Lime green color
    # Draw Booklet Number bubbles
    draw_label_group(result_image, template_data["booklet_no"], anchor_x, anchor_y, (255, 100, 0), "BOOK_") # Light blue/Cyan color


    # --- Step 6: Display the result image ---
    print("\n[INFO] Displaying final result image with detected regions. (Check your Colab output)")
    cv2.imshow("Result Image", result_image) # Display image in Google Colab
    cv2.waitKey(0) # Wait for a key press to close the window
    cv2.destroyAllWindows() # Close all OpenCV windows

    # --- Step 7: Save detected answers to a JSON file ---
    output_answers_json = os.path.join(temp_dir, f"{test_name}_detected_answers.json")
    with open(output_answers_json, "w") as f:
        json.dump(detected_answers, f, indent=4) # Save with indentation for readability
    print(f"[INFO] Detected answers saved to: {output_answers_json}")

    return detected_answers # Return the dictionary of detected answers


if __name__ == '__main__':
    # --- Example Usage (for testing omr_processor.py directly) ---
    print("--- Running omr_processor.py as main script ---")

    # Define paths for dummy files (replace with your actual paths for real use)
    _temp_dir = "./OMR_DEV/temp"
    _template_img_path = os.path.join(_temp_dir, "D:/OMR_DEV/dataset/BE24/images/be4d08e9-BE24-05-01001.jpg")
    _label_file_path = os.path.join(_temp_dir, "D:/OMR_DEV/dataset/BE24/labels/be4d08e9-BE24-05-01001.txt")
    _classes_file_path = os.path.join(_temp_dir, "D:/OMR_DEV/dataset/BE24/classes.txt")
    _test_img_path = os.path.join(_temp_dir, "D:/OMR_DEV/OMR/OMR-Assam/BE24-05-06/BE24-05-06001.jpg")

    # --- Process the OMR sheet using the main function ---
    detected_data_output = process_omr_sheet(
        test_image_path=_test_img_path,
        template_image_path=_template_img_path,
        label_file_path=_label_file_path,
        classes_file_path=_classes_file_path,
        anchor_name="anchor_1",
        temp_dir=_temp_dir
    )
    print("\n[SUCCESS] OMR processing complete. Final Detected Data (returned by function):")
    print(detected_data_output)