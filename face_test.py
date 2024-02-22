import cv2
from typing import Tuple, Union
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  annotated_image = image.copy()
  height, width, _ = image.shape

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

    # Draw keypoints
    for keypoint in detection.keypoints:
      keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                     width, height)
      color, thickness, radius = (0, 255, 0), 2, 2
      cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    category_name = '' if category_name is None else category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return annotated_image

class BoundingBox:
    def __init__(self, origin_x, origin_y, width, height):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width
        self.height = height

    def to_xyxy(self):
        x_min = self.origin_x
        y_min = self.origin_y
        x_max = self.origin_x + self.width
        y_max = self.origin_y + self.height
        return x_min, y_min, x_max, y_max
    
    def box_touches_edge(self, image_width, image_height):
        x_min, y_min, x_max, y_max = self.to_xyxy()
        # Check if any side of the bounding box intersects with the edges of the image
        if x_min <= 0 or x_max >= image_width or y_min <= 0 or y_max >= image_height:
            return True
        else:
            return False

# IMAGE_FILE = "test5.jpg"
# IMAGE_FILE = "test.png"
IMAGE_FILE = "test3.jpg"

BRIGHTNESS_THRESHOLD = 60
BLUR_THRESHOLD = 100

# STEP 2: Create an FaceDetector object.
base_options = python.BaseOptions(model_asset_path='detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file(IMAGE_FILE)
img = cv2.imread(IMAGE_FILE)
# STEP 4: Detect faces in the input image.
detection_result = detector.detect(image)
print(len(detection_result.detections))
if len(detection_result.detections) > 0:
    # STEP 5: Process the detection result. In this case, visualize it.
    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    image_height, image_width = image_copy.shape[0], image_copy.shape[1]
    d_bbox = detection_result.detections[0].bounding_box
    bbox = BoundingBox(origin_x=d_bbox.origin_x, origin_y=d_bbox.origin_y, width=d_bbox.width, height=d_bbox.height)
    bbox.box_touches_edge(image_width=image_width, image_height=image_height)

    cv2.imwrite("image2.jpg",rgb_annotated_image,)

    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    # Calculate the average pixel intensity
    average_intensity = cv2.mean(gray_image)[0]
    print("avg", average_intensity)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    avg = cv2.mean(hsv)[1]
    print("h avg", avg)

# Average focus measure across the image
    # average_focus = blur_map.mean()
    # print("focus:", average_focus)
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()

    print("Variance of Laplacian:", laplacian_var)