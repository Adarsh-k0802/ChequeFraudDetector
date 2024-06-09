import streamlit as st
import cv2
from ultralytics import YOLO  # Assuming ultralytics is installed
import numpy as np
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
import os
from PIL import Image, ImageDraw, ImageFont
import glob
import re
import json


# Initialize YOLO model
model = YOLO('latest.pt')

# Replace with your subscription key and endpoint
key = " "
endpoint = ""

document_intelligence_client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))


def ensure_azure_credentials():
    """
    Checks if Azure Document Intelligence credentials are set and warns if not.
    """
    if not key or not endpoint:
        st.warning("Azure Document Intelligence credentials not set! Please provide subscription key and endpoint.")
        return False
    return True


def extract_text_from_image(image):
    """
    Extracts text from an image using Azure Document Intelligence.

    Args:
        image: The image (BGR format) to extract text from.

    Returns:
        A dictionary containing extracted text information or an error message.
    """

    if not ensure_azure_credentials():
        return {"error": "Azure Document Intelligence credentials not set!"}

    try:
        # Convert BGR to RGB (expected by Document Intelligence)
        # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



        with open(image, "rb") as f:
            poller = document_intelligence_client.begin_analyze_document(
                "prebuilt-layout", analyze_request=f, content_type="application/octet-stream"
            )
        result: AnalyzeResult = poller.result()
        # print(f"Text\n{result.content}")

        # st.write(result.content)
        return result.content

    except Exception as e:
        return {"error": f"Error extracting text: {e}"}


def perform_object_detection(image_path):
    """
    Performs object detection with YOLO and displays results.
    """

    # Perform object detection
    result = model.predict(image_path)

    # Plot results
    processed_image = result[0].plot()


   

    # Display annotated image
    st.image(processed_image, channels="BGR", use_column_width=True)

    # Save annotated image (optional)
    result[0].save_crop("crop")


def create_output_image(crop_folder_path):
    max_width = 650  # Maximum width for input images
    margin = 10  # Margin between headings and images
    extra_bottom_margin = 60  # Extra bottom margin for each image

    total_height = 0
    images = []

    specific_folders = ["AmountDigit", "AmountText", "Bank_name", "Branch_address", "Date", "PayTo", "a-c_number", "cheque_number", "IFSC"]

    for folder in specific_folders:
        folder_dir = os.path.join(crop_folder_path, folder)

        if not os.path.isdir(folder_dir):
            continue

        image_files = glob.glob(os.path.join(folder_dir, '*.jpg')) + glob.glob(os.path.join(folder_dir, '*.png'))
        if not image_files:
            continue

        # Add heading for the folder
        total_height += 30 + margin  # Heading height + margin
        images.append((f"{folder}", None))

        # Add images for the folder
        for img_path in image_files:
            img = Image.open(img_path)
            img_width, img_height = img.size

            # Calculate the new dimensions to fit inside the result image
            if img_width > max_width:
                new_width = max_width
                new_height = int(img_height * (max_width / img_width))
                img = img.resize((new_width, new_height))
                img_width, img_height = new_width, new_height

            images.append((img, (img_width, img_height)))

            # Add margin after each image
            total_height += img_height + margin + extra_bottom_margin

            # Stop after processing the first image in the "Branch_address" folder
            if folder == "Branch_address":
                break
            if folder == "IFSC":
                break
            if folder == "AmountText":
                break
            if folder == "PayTo":
                break
            if folder == "Bank_name":
                break
    # Create a new image with the calculated dimensions
    result = Image.new('RGB', (max_width, total_height), 'white')
    draw = ImageDraw.Draw(result)
    font = ImageFont.truetype("arial.ttf", 20)

    y_offset = 0  # Start with top margin

    for item in images:
        if isinstance(item[0], str):  # Handle headings
            draw.text((10, y_offset), item[0], fill='black', font=font)
            y_offset += 30 + margin  # Heading height + margin
        elif isinstance(item[0], Image.Image):  # Handle images
            x_offset = 0  # Left align images
            result.paste(item[0], (x_offset, y_offset))
            y_offset += item[1][1] + margin + extra_bottom_margin  # Image height + margin + extra bottom margin

    # Save the output image as "output.jpg"
    result.save("output.jpg")

def create_empty_json():
    # Define the keys in the specified order
    keys = ["AmountDigit", "AmountText", "Bank_name", "Branch_address", "Date", "PayTo", "a-c_number", "cheque_number", "IFSC"]
    
    # Create an empty dictionary with the keys
    empty_json = {key: "" for key in keys}
    
    return empty_json


def preprocess_text(text):
    text = re.sub(r"\s+", " ", text).strip()

    patterns = [
        (r"amount\stext", "AmountText"),
        (r"amount\sdigit", "AmountDigit"),
        (r"amounttext", "AmountText"),
        (r"amountdigit", "AmountDigit"),
        (r"bank\sname", "Bank_name"),
        (r"bankname", "Bank_name"),
        (r"IFSC\s(\w+)", r"IFSC:\1"), 
        (r"IFSC-(\w+)", r"IFSC:\1"),
        (r"IFSC\s-\s(\w+)", r"IFSC:\1"),
        (r"IFSC-\s(\w+)", r"IFSC:\1"),
        (r"IFSC\s-(\w+)", r"IFSC:\1"),
        (r"crome", "crore"),
        (r"Crome", "Crore"),
        (r"/", "1"),
        (r"lash", "lakh"),
        (r"laves", "lakhs"),
    ]

    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text

def format_date(date_str):
    # Assuming date_str is in the format DDMMYYYY
    if len(date_str) == 8:
        day = date_str[:2]
        month = date_str[2:4]
        year = date_str[4:]
        formatted_date = f"{day}-{month}-{year}"
        return formatted_date
    else:
        return date_str

def process_cheque_number(cheque_number_value):
    # Remove the last digit if it's "1"
    if cheque_number_value and cheque_number_value[-1] == "1":
        return cheque_number_value[:-1]
    return cheque_number_value

def process_amount_digit(amount_digit_value):
    # Remove the last digit if it's "1"
    if amount_digit_value and amount_digit_value[-1] == "1":
        return amount_digit_value[:-1]
    return amount_digit_value

def populate_json_from_text(text):
    # Create an empty JSON object with specified keys
    json_object = create_empty_json()  # Assuming you have a function to create an empty JSON object

    # Define the keys in the order they appear in the text
    keys = list(json_object.keys())

    # Remove extra spaces and ensure single space between words
    cleaned_text = re.sub(r'\s+', ' ', text)

    # Iterate over the keys and extract text between them
    for i in range(len(keys) - 1):
        start_key = keys[i]
        end_key = keys[i + 1]
        pattern = re.escape(start_key) + r"(.*?)\s*" + re.escape(end_key)
        match = re.search(pattern, cleaned_text)
        if match:
            value = match.group(1).strip()
            if start_key == "Date":
                value = format_date(value)  # Format the date value
            elif start_key == "cheque_number":
                value = process_cheque_number(value) 
            elif start_key == "AmountDigit":
                value = process_amount_digit(value)  # Process the AmountDigit value
            json_object[start_key] = value


    # Extract IFSC separately because it doesn't have a following key
    if "IFSC" in cleaned_text:
        match = re.search(r"IFSC\s*:\s*([A-Za-z0-9]+)", cleaned_text)
        if match:
            json_object["IFSC"] = match.group(1).strip()

    return json_object


def main():
    st.title('Cheque Attribute Extraction')
    st.write('Upload a Cheque and see extracted attributes.')


    if not ensure_azure_credentials():
        return

    # File uploader
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "png"])

    if uploaded_file is not None:
        # Read uploaded image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform object detection with YOLO
        perform_object_detection(image)
        crop_folder_path ="crop"
        create_output_image(crop_folder_path)
        

        text_path = extract_text_from_image("output.jpg")
        fprocessed_text = preprocess_text(text_path)
        json_data = populate_json_from_text(fprocessed_text)
        st.code(json.dumps(json_data, indent=4),language="json")




# Run the Streamlit app
if __name__ == '__main__':
    main()
