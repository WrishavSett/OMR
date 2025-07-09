import os
import re
import cv2
import json
import numpy as np
import pandas as pd

temp= "./temp"
if not os.path.exists(temp):
    os.makedirs(temp)

template_image_path = "D:/OMR_DEV/T2/images/16dce875-TEST-01003.jpg"
label_file_path = "D:/OMR_DEV/T2/labels/16dce875-TEST-01003.txt"
class_file_path = "D:/OMR_DEV/T2/classes.txt"

# Load the image
template_image = cv2.imread(template_image_path)

# Load the labels (assuming YOLO format)
labels = []
with open(label_file_path, 'r') as f:
    for line in f:
        labels.append(line.strip().split())

# Load the class names
classes = []
with open(class_file_path, 'r') as f:
    for line in f:
        classes.append(line.strip())

print("[INFO] Image loaded successfully.")
print("[INFO] Labels loaded successfully.")
print("[INFO] Classes loaded successfully:", classes)
print("[INFO] Total number of classes:", len(classes))

# Identify anchor boxes from the labels and extract their coordinates
anchor_boxes = {}
image_height, image_width = template_image.shape[:2]

for label in labels:
    class_id, x_center_norm, y_center_norm, width_norm, height_norm = label
    class_name = classes[int(class_id)]

    if "anchor" in class_name:
        # Convert normalized coordinates to pixel coordinates
        x_center = float(x_center_norm) * image_width
        y_center = float(y_center_norm) * image_height
        width = float(width_norm) * image_width
        height = float(height_norm) * image_height

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        center_x = int(x_center)
        center_y = int(y_center)

        anchor_boxes[class_name] = {
            "bounding_box": (x1, y1, x2, y2),
            "center": (center_x, center_y)
        }

# Draw the anchor boxes and their center points on the image
image_with_anchors = template_image.copy()

for class_name, coords in anchor_boxes.items():
    x1, y1, x2, y2 = coords["bounding_box"]
    center_x, center_y = coords["center"]

    # Draw rectangle
    cv2.rectangle(image_with_anchors, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image_with_anchors, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw center
    cv2.circle(image_with_anchors, (center_x, center_y), 5, (0, 0, 255), -1)

    # Print the coordinates
    print(f"[INFO] {class_name} Bounding Box: ({x1}, {y1}, {x2}, {y2})")
    print(f"[INFO] {class_name} Center Point: ({center_x}, {center_y})")

cv2.imshow("Anchor points", image_with_anchors)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------- Step 2: Parse bounding boxes and centers ----------
object_centers = {}
object_boxes = {}

for label in labels:
    class_id, x_c, y_c, w, h = map(float, label)
    class_name = classes[int(class_id)]

    x_center = x_c * image_width
    y_center = y_c * image_height
    width = w * image_width
    height = h * image_height

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    object_centers[class_name] = (x_center, y_center)
    object_boxes[class_name] = (x1, y1, x2, y2)

# ---------- Step 3: Extract anchor_1 as reference ----------
anchor_name = "anchor_1"
anchor_center = object_centers[anchor_name]

# ---------- Step 4: Build relative data structure ----------
def parse_class_name(name):
    import re

    # Questions and options
    if re.match(r'^question_\d+$', name):
        return "question", int(name.split('_')[1])
    elif re.match(r'^\d+[A-D]$', name):
        return "option", int(re.match(r'^(\d+)', name).group(1))

    # Registration number characters
    elif name.startswith("reg_no") and name != "reg_no":
        return "reg_no_char", name
    elif name == "reg_no":
        return "reg_no_main", name

    # Roll number characters
    elif name.startswith("roll_no") and name != "roll_no":
        return "roll_no_char", name
    elif name == "roll_no":
        return "roll_no_main", name

    # Booklet number characters
    elif name.startswith("booklet_no") and name != "booklet_no":
        return "booklet_no_char", name
    elif name == "booklet_no":
        return "booklet_no_main", name

    else:
        return None, None

question_data = {}
question_data = {
    "questions": {},
    "reg_no": {},
    "roll_no": {},
    "booklet_no": {}
}

for name in object_centers:
    if name == anchor_name:
        continue

    kind, identifier = parse_class_name(name)
    if kind is None:
        continue

    cx, cy = object_centers[name]
    x1, y1, x2, y2 = object_boxes[name]

    rel = {
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

    if kind == "question":
        qnum = identifier
        if qnum not in question_data["questions"]:
            question_data["questions"][qnum] = { "question": {}, "options": {} }
        question_data["questions"][qnum]["question"] = rel
    elif kind == "option":
        qnum = identifier
        if qnum not in question_data["questions"]:
            question_data["questions"][qnum] = { "question": {}, "options": {} }
        question_data["questions"][qnum]["options"][name] = rel
    elif kind in ["reg_no_char", "reg_no_main"]:
        question_data["reg_no"][identifier] = rel
    elif kind in ["roll_no_char", "roll_no_main"]:
        question_data["roll_no"][identifier] = rel
    elif kind in ["booklet_no_char", "booklet_no_main"]:
        question_data["booklet_no"][identifier] = rel

# ---------- Step 5: Save relative structure to JSON ----------
output_json = f"{temp}/question_relative_positions.json"
with open(output_json, "w") as f:
    json.dump(question_data, f, indent=2)

print(f"[INFO] Saved relative data to: {output_json}")

# Test image details

test_image_path = "D:/OMR_DEV/OMR/TEST/TEST-01010.jpg"
test_image = cv2.imread(test_image_path)
cv2.imshow("Test Image", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------- Step 2: Find anchor_1 using ORB + Homography ----------
orb = cv2.ORB_create(5000)
kp1, des1 = orb.detectAndCompute(template_image, None)
kp2, des2 = orb.detectAndCompute(test_image, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.knnMatch(des1, des2, k=2)

# Ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

print(f"[INFO] Found {len(good)} good matches")

# Homography
if len(good) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    anchor_pt = np.array([[object_centers[anchor_name]]], dtype=np.float32)
    transformed_anchor = cv2.perspectiveTransform(anchor_pt, M)
    transformed_center = tuple(map(int, transformed_anchor[0][0]))
    print(f"[INFO] Anchor_1 in test image: {transformed_center}")
else:
    raise Exception("[ERROR] Not enough good matches found for homography")

# ---------- Step 3: Load relative question data ----------
with open(f"{temp}/question_relative_positions.json", "r") as f:
    question_data = json.load(f)

anchor_x, anchor_y = transformed_center
result_image = test_image.copy()

# ---------- Step 4: Draw centers and bounding boxes ----------
for qnum, qdata in question_data["questions"].items():
    qnum = int(qnum)

    # Question
    if "question" in qdata:
        qrel = qdata["question"]
        dx, dy = qrel["center"]["dx"], qrel["center"]["dy"]
        cx, cy = int(anchor_x + dx), int(anchor_y + dy)

        cv2.circle(result_image, (cx, cy), 4, (255, 0, 255), -1)
        cv2.putText(result_image, f"Q{qnum}", (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        if "bbox" in qrel:
            x1 = int(anchor_x + qrel["bbox"]["x1"])
            y1 = int(anchor_y + qrel["bbox"]["y1"])
            x2 = int(anchor_x + qrel["bbox"]["x2"])
            y2 = int(anchor_y + qrel["bbox"]["y2"])
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 255), 2)

    # Options
    for opt_name, opt_data in qdata.get("options", {}).items():
        dx, dy = opt_data["center"]["dx"], opt_data["center"]["dy"]
        ox, oy = int(anchor_x + dx), int(anchor_y + dy)

        cv2.circle(result_image, (ox, oy), 4, (0, 255, 255), -1)
        cv2.putText(result_image, opt_name, (ox + 5, oy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if "bbox" in opt_data:
            x1 = int(anchor_x + opt_data["bbox"]["x1"])
            y1 = int(anchor_y + opt_data["bbox"]["y1"])
            x2 = int(anchor_x + opt_data["bbox"]["x2"])
            y2 = int(anchor_y + opt_data["bbox"]["y2"])
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

# Generalized visualization
def draw_label_group(label_group, color, label_prefix=""):
    for name, data in label_group.items():
        dx, dy = data["center"]["dx"], data["center"]["dy"]
        cx, cy = int(anchor_x + dx), int(anchor_y + dy)

        cv2.circle(result_image, (cx, cy), 4, color, -1)
        cv2.putText(result_image, f"{label_prefix}{name}", (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        x1 = int(anchor_x + data["bbox"]["x1"])
        y1 = int(anchor_y + data["bbox"]["y1"])
        x2 = int(anchor_x + data["bbox"]["x2"])
        y2 = int(anchor_y + data["bbox"]["y2"])
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 1)

draw_label_group(question_data["reg_no"], (0, 100, 255), "REG_")
draw_label_group(question_data["roll_no"], (100, 255, 0), "ROLL_")
draw_label_group(question_data["booklet_no"], (255, 100, 0), "BOOK_")

# ---------- Step 5: Display result ----------
cv2.imshow("Result", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------- Step 6: Detect filled options and digit fields ----------
marked_options = {}
reg_digits = {}
roll_digits = {}
booklet_digits = {}

fill_threshold = 0.67  # Threshold for filled circle detection

# ------- Detect marked options -------
for qnum, qdata in question_data["questions"].items():
    qnum = int(qnum)
    marked_options[qnum] = []

    for opt_name, opt_data in qdata.get("options", {}).items():
        x1 = int(anchor_x + opt_data["bbox"]["x1"])
        y1 = int(anchor_y + opt_data["bbox"]["y1"])
        x2 = int(anchor_x + opt_data["bbox"]["x2"])
        y2 = int(anchor_y + opt_data["bbox"]["y2"])

        # Clip to bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(test_image.shape[1] - 1, x2), min(test_image.shape[0] - 1, y2)

        roi = test_image[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fill_ratio = np.sum(binary == 0) / binary.size

        print(f"Q{qnum} - {opt_name} Fill Ratio: {fill_ratio:.2f}")
        cv2.putText(result_image, f"{fill_ratio:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        if fill_ratio < fill_threshold:
            marked_options[qnum].append(opt_name)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# ------- Helper: Extract digits -------
def detect_digits_from_group(group_data, group_name):
    digit_map = {}
    for name, data in group_data.items():
        match = re.match(rf"{group_name}_(\d+)_(\d+)", name)
        if not match:
            continue
        index, digit = map(int, match.groups())

        x1 = int(anchor_x + data["bbox"]["x1"])
        y1 = int(anchor_y + data["bbox"]["y1"])
        x2 = int(anchor_x + data["bbox"]["x2"])
        y2 = int(anchor_y + data["bbox"]["y2"])

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(test_image.shape[1] - 1, x2), min(test_image.shape[0] - 1, y2)

        roi = test_image[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fill_ratio = np.sum(binary == 0) / binary.size

        cv2.putText(result_image, f"{fill_ratio:.2f}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 0, 255), 1)

        if fill_ratio < fill_threshold:
            if index not in digit_map:
                digit_map[index] = []
            digit_map[index].append((digit, fill_ratio))

    resolved = {}
    for idx in sorted(digit_map):
        digits = sorted(digit_map[idx], key=lambda x: x[1])  # Most filled first
        resolved[idx] = digits[0][0]
    return resolved

# ------- Detect digits -------
reg_digits = detect_digits_from_group(question_data["reg_no"], "reg_no")
roll_digits = detect_digits_from_group(question_data["roll_no"], "roll_no")
booklet_digits = detect_digits_from_group(question_data["booklet_no"], "booklet_no")

# ------- Format as strings -------
def format_digits(d):
    return ''.join(str(d[i]) for i in sorted(d)) if d else "NA"

reg_str = format_digits(reg_digits)
roll_str = format_digits(roll_digits)
booklet_str = format_digits(booklet_digits)

# ---------- Step 7: Show and Save Final Output ----------
print("\n[INFO] Detected Marked Options:")
for q, opts in marked_options.items():
    print(f"Q{q}: {opts}")

print("\n[INFO] Detected Numbers:")
print("[INFO] Registration No. :", reg_str)
print("[INFO] Roll No.         :", roll_str)
print("[INFO] Booklet No.      :", booklet_str)

cv2.imshow("Result", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save combined output
final_output = {
    "marked_options": marked_options,
    "registration_number": reg_str,
    "roll_number": roll_str,
    "booklet_number": booklet_str
}

with open(f"{temp}/marked_answers_and_ids.json", "w") as f:
    json.dump(final_output, f, indent=2)

print(f"\n[INFO] Saved output to '{temp}/marked_answers_and_ids.json'")