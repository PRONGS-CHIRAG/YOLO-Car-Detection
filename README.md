# YOLO-Car-Detection


## Overview
This project implements an autonomous driving application that detects cars in images using the **YOLO (You Only Look Once) object detection model**. The model processes images and predicts bounding boxes around detected vehicles, making it useful for real-time object detection in autonomous navigation systems.

## Features
- Utilizes **YOLOv2** for real-time object detection.
- Processes images to identify and localize vehicles.
- Implements **Non-Maximum Suppression (NMS)** to filter overlapping bounding boxes.
- Uses TensorFlow and Keras for deep learning inference.
- Includes visualization utilities for displaying predictions.

## Project Structure
- **`Autonomous_driving_application_Car_detection.ipynb`**: The main Jupyter Notebook containing step-by-step implementation of YOLO-based car detection.
- **`utils.py`**: Utility functions for:
  - Preprocessing images for YOLO.
  - Filtering predictions based on confidence scores.
  - Applying Non-Maximum Suppression (NMS) to remove redundant bounding boxes.
  - Drawing bounding boxes on detected objects.
  - Evaluating the YOLO model output and making predictions.

## Dependencies
To run this project, install the following packages:
```bash
pip install tensorflow numpy pandas scipy pillow matplotlib
```

## How to Run
1. Clone this repository and navigate to the project directory.
2. In the repository clone the repository https://github.com/allanzelener/YAD2K/tree/master and move the files and subfolders to the main project directory.
3. Open `Autonomous_driving_application_Car_detection.ipynb` in Jupyter Notebook.
4. Run all cells to:
   - Load the YOLO model.
   - Preprocess images for detection.
   - Predict car locations.
   - Display the output with bounding boxes.

## YOLO Model Details
- **Input**: Image of size (608, 608, 3) normalized for YOLO processing.
- **Output**: Predicted bounding boxes with class probabilities.
- **Filtering**:
  - Boxes with confidence scores below `threshold=0.6` are discarded.
  - **Intersection over Union (IoU)** used to filter overlapping boxes (`iou_threshold=0.5`).
  - Maximum of 10 bounding boxes per image.

## Results
The model successfully detects vehicles in images, outputs bounding boxes, and applies Non-Maximum Suppression to refine predictions. The detected objects are drawn on the images for visualization.

## Future Improvements
- Train on additional datasets for improved accuracy.
- Implement real-time video detection.
- Optimize model for deployment on edge devices (e.g., Raspberry Pi, Jetson Nano).

## Credits
This project is inspired by deep learning applications in autonomous driving and object detection. The implementation is based on YOLO principles.

## License
This project is open-source and available under the MIT License.

---
**Author:** Chirag N Vijay  

