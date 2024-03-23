
import cv2
import streamlit as st
import numpy as np
from PIL import Image




# Function to brighten the image
def brighten_image(image, amount):
    img_bright = cv2.convertScaleAbs(image, beta=amount)
    return img_bright

# Function to apply blur to the image
def blur_image(image, amount):
    blur_img = cv2.GaussianBlur(image, (11, 11), amount)
    return blur_img

# Function to enhance details in the image
def enhance_details(img):
    hdr = cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15)
    return hdr

# Function to convert PIL image to OpenCV image
def pil_to_cv2(image):
    image_cv2 = np.array(image)
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)
    return image_cv2

def detect_objects(image):
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers_indices = net.getUnconnectedOutLayers()

    if len(layer_names) == 0 or len(output_layers_indices) == 0:
        return image # Return the original image if layer names or output layer indices are empty

    output_layers = [layer_names[i - 1] for i in output_layers_indices]

    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return image




# def count_vehicles(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)
#     contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Filter contours based on area (vehicle size)
#     min_area = 1000  # Adjust this threshold according to your images
#     max_area = 15000  # Adjust this threshold according to your images
#     vehicle_count = 0
#     # Example of filtering based on aspect ratio
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if min_area < area < max_area:
#             vehicle_count += 1
    
#     return vehicle_count

def count_vehicles(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply thresholding
    _, thresh = cv2.threshold(blurred, 177, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on size and shape
    vehicle_count = 0
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)
        # Calculate minimum enclosing circle radius
        (x, y), radius = cv2.minEnclosingCircle(contour)
        
        # Filter based on area and radius
        if 500 < area < 50000 and 10 < radius < 50:
            vehicle_count += 1
    
    return vehicle_count


# Main function to create the Streamlit app
def main_loop():
    st.title("OpenCV Demo App")
    st.subheader("This app allows you to play with Image filters and Object Detection!")
    st.text("We use OpenCV and Streamlit for this demo")

    blur_rate = st.sidebar.slider("Blurring", min_value=0.5, max_value=3.5)
    brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)
    apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')
    detect_objects_flag = st.sidebar.checkbox('Detect Objects')

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)

    # Convert PIL image to OpenCV image
    original_image_cv2 = pil_to_cv2(original_image)

    # Apply image processing filters
    processed_image = blur_image(original_image_cv2, blur_rate)
    processed_image = brighten_image(processed_image, brightness_amount)
    if apply_enhancement_filter:
        processed_image = enhance_details(processed_image)

    # Perform object detection (detect vehicles) if the option is enabled
    if detect_objects_flag:
        processed_image = detect_objects(processed_image)
        # Count vehicles using contour detection
        vehicle_count = count_vehicles(processed_image)
        # Display vehicle count
        st.write("Number of Vehicles Detected:", vehicle_count)

    # Display original and processed images
    st.text("Original Image vs Processed Image")
    st.image([original_image, processed_image], use_column_width=True)

# Run the main loop
if __name__ == '__main__':
    main_loop()

