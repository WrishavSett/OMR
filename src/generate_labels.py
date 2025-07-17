def generate_labels():
    """
    Queries the user for template details and generates a text file with labels.
    """
    print("--- Label Generator ---")

    # 1. Query for template name
    template_name = input("Enter the template name: ").strip()
    if not template_name:
        print("Template name cannot be empty. Exiting.")
        return

    # 2. Query for number of anchors
    num_anchors = 0
    while True:
        try:
            num_anchors_str = input("Enter the number of anchors to generate (e.g., 4 for anchor_1 to anchor_4): ").strip()
            if not num_anchors_str:
                print("Number of anchors cannot be empty. Please enter a number.")
                continue
            num_anchors = int(num_anchors_str)
            if num_anchors < 0:
                print("Number of anchors must be a non-negative integer.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a valid integer for the number of anchors.")

    # 3. Query for key fields
    available_key_fields = ["omr_sheet_no", "reg_no", "roll_no", "booklet_no"]
    selected_key_fields = []
    # Dictionary to store the number of sub-fields and sub-sub-fields for each selected key field
    key_field_details = {}

    print("\nAvailable key fields: omr_sheet_no, reg_no, roll_no, booklet_no")
    print("Enter key fields separated by commas (e.g., reg_no, booklet_no). Press Enter to skip.")

    while True:
        user_input_fields = input("Selected key fields: ").strip().lower()
        if not user_input_fields:
            break
        
        input_fields_list = [field.strip() for field in user_input_fields.split(',')]
        
        valid_input = True
        for field in input_fields_list:
            if field not in available_key_fields:
                print(f"'{field}' is not a valid key field. Please choose from: {', '.join(available_key_fields)}")
                valid_input = False
                break
            if field in selected_key_fields:
                print(f"'{field}' has already been added. Please enter unique fields.")
                valid_input = False
                break
        
        if valid_input:
            selected_key_fields.extend(input_fields_list)
            break # Exit loop if valid input is provided

    # Query for sub-field details for each selected key field (excluding omr_sheet_no)
    for field in selected_key_fields:
        if field != "omr_sheet_no":
            print(f"\n--- Details for '{field}' ---")
            num_sub_fields = 0
            while True:
                try:
                    num_sub_fields_str = input(f"Enter the number of top-level sub-fields for '{field}' (e.g., 10 for {field}_0 to {field}_9): ").strip()
                    if not num_sub_fields_str:
                        print("Number of sub-fields cannot be empty. Please enter a number.")
                        continue
                    num_sub_fields = int(num_sub_fields_str)
                    if num_sub_fields < 0:
                        print("Number of sub-fields must be a non-negative integer.")
                    else:
                        break
                except ValueError:
                    print("Invalid input. Please enter a valid integer.")
            
            num_sub_sub_fields = 0
            if num_sub_fields > 0: # Only ask for sub-sub-fields if there are sub-fields
                while True:
                    try:
                        num_sub_sub_fields_str = input(f"Enter the number of sub-sub-fields for each '{field}_X' (e.g., 10 for {field}_X_0 to {field}_X_9): ").strip()
                        if not num_sub_sub_fields_str:
                            print("Number of sub-sub-fields cannot be empty. Please enter a number.")
                            continue
                        num_sub_sub_fields = int(num_sub_sub_fields_str)
                        if num_sub_sub_fields < 0:
                            print("Number of sub-sub-fields must be a non-negative integer.")
                        else:
                            break
                    except ValueError:
                        print("Invalid input. Please enter a valid integer.")
            
            key_field_details[field] = {
                "num_sub_fields": num_sub_fields,
                "num_sub_sub_fields": num_sub_sub_fields
            }

    # Query for number of questions and options per question
    num_questions = 0
    while True:
        try:
            num_questions_str = input("Enter the number of questions: ").strip()
            if not num_questions_str:
                print("Number of questions cannot be empty. Please enter a number.")
                continue
            num_questions = int(num_questions_str)
            if num_questions <= 0:
                print("Number of questions must be a positive integer.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a valid integer for the number of questions.")

    options_per_question = 0
    while True:
        try:
            options_per_question_str = input("Enter the number of options per question (e.g., 4 for A, B, C, D): ").strip()
            if not options_per_question_str:
                print("Number of options cannot be empty. Please enter a number.")
                continue
            options_per_question = int(options_per_question_str)
            if options_per_question <= 0:
                print("Number of options must be a positive integer.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a valid integer for options per question.")

    # Generate labels
    labels = []
    for i in range(1, num_anchors + 1):
        labels.append(f"anchor_{i}")

    # Add selected key fields and their sub-labels based on user input
    for field in selected_key_fields:
        labels.append(field)
        # Special handling for omr_sheet_no: no child entries
        if field == "omr_sheet_no":
            continue # Skip adding sub-labels for omr_sheet_no
        
        details = key_field_details.get(field, {"num_sub_fields": 0, "num_sub_sub_fields": 0})
        num_sub_fields = details["num_sub_fields"]
        num_sub_sub_fields = details["num_sub_sub_fields"]

        for i in range(num_sub_fields):
            labels.append(f"{field}_{i}")
            for j in range(num_sub_sub_fields):
                labels.append(f"{field}_{i}_{j}")

    # Add questions and options
    for i in range(1, num_questions + 1):
        labels.append(f"question_{i}")
        for j in range(options_per_question):
            # Assuming options are A, B, C, D, etc.
            option_char = chr(ord('A') + j)
            labels.append(f"{i}{option_char}")

    # Save to file
    file_name = f"{template_name}.txt"
    try:
        with open(file_name, "w") as f:
            for label in labels:
                f.write(label + "\n")
        print(f"\nLabels successfully generated and saved to '{file_name}'")
    except IOError as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    generate_labels()
