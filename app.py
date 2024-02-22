from flask import Flask, request, jsonify
import cv2
from typing import Tuple, Union
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = Flask(__name__)

# Thresholds for brightness and confidence
BRIGHTNESS_THRESHOLD = 100
CONF_THRESHOLD = 100

# Constants for text display
MARGIN = 10
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # Red

# Helper function to convert normalized coordinates to pixel coordinates
def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  
  # Check if the normalized value is valid
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

  # Convert normalized coordinates to pixel coordinates
  if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

# Function to visualize bounding boxes and keypoints on the input image
def visualize(
    image,
    detection_result
) -> np.ndarray:
  annotated_image = image.copy()  # Make a copy of the input image
  height, width, _ = image.shape  # Get dimensions of the image

  # Loop through each detection
  for detection in detection_result.detections:
    # Draw bounding box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

    # Draw keypoints
    for keypoint in detection.keypoints:
      keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height)
      color, thickness, radius = (0, 255, 0), 2, 2
      cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    category_name = '' if category_name is None else category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return annotated_image

# Class representing a bounding box
class BoundingBox:
    def __init__(self, origin_x, origin_y, width, height):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width
        self.height = height

    # Convert bounding box coordinates to x1, y1, x2, y2 format
    def to_xyxy(self):
        x_min = self.origin_x
        y_min = self.origin_y
        x_max = self.origin_x + self.width
        y_max = self.origin_y + self.height
        return x_min, y_min, x_max, y_max
    
    # Check if bounding box touches the edge of the image
    def box_touches_edge(self, image_width, image_height):
        x_min, y_min, x_max, y_max = self.to_xyxy()
        # Check if any side of the bounding box intersects with the edges of the image
        if x_min <= 0 or x_max >= image_width or y_min <= 0 or y_max >= image_height:
            return True
        else:
            return False

# Route for processing image
@app.route('/process_image', methods=['POST'])
def process_image():
    # Receive image file
    file = request.files['image']
    nparr = np.frombuffer(file.read(), np.uint8)
    in_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(in_image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image_rgb))
    # Create a FaceDetector object
    base_options = python.BaseOptions(model_asset_path='detector.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    # Detect faces in the image
    detection_result = detector.detect(mp_image)
    face_count = len(detection_result.detections)
    np_image = np.copy(mp_image.numpy_view())

    # If faces are detected, visualize the result
    if face_count > 0:
        # Get image dimensions and bounding box information
        image_height, image_width = np_image.shape[0], np_image.shape[1]
        d_bbox = detection_result.detections[0].bounding_box
        bbox = BoundingBox(origin_x=d_bbox.origin_x, origin_y=d_bbox.origin_y, width=d_bbox.width, height=d_bbox.height)
        is_cropped = bbox.box_touches_edge(image_width=image_width, image_height=image_height)
        conf_score = detection_result.detections[0].categories[0].score
    else:
        is_cropped = False
        conf_score = 0.0

    # Convert the image to grayscale and calculate average luminance
    gray_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(np_image, cv2.COLOR_BGR2HLS)
    lum_avg = cv2.mean(hsv)[1]

    # Calculate variance of Laplacian
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()

    # Print results
    print("Face count:", face_count)
    print("Face is cropped:", is_cropped)
    print("Confidence score:", conf_score)
    print("Luminance average:", lum_avg)
    print("Variance of Laplacian:", laplacian_var)

    # Check if the image meets the criteria
    is_ok = True
    error = ''

    if face_count != 1:
        is_ok = False
        error += f'Face count is {face_count}. ' 

    if is_cropped == True:
        is_ok = False
        error += 'Face is cropped. '

    if lum_avg < 100:
        is_ok = False
        error += "Image is too dark. "  

    if conf_score <= 0.85:
        is_ok = False
        error += "Face is not clear. "

    # Print final assessment
    print(f"Picture is ok: {is_ok}")
    print(f"Error: {error}")
    # Further processing and assessment

    # Return assessment results
    return jsonify({
        'is_ok': is_ok,
        'error': error,
        'conf': conf_score
        # Add other assessment results here
    })

if __name__ == '__main__':
    app.run(debug=True)