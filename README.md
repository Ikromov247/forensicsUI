Forensics UI - Video Similarity Search

This project implements a video similarity search application designed for forensic investigations. The application takes an image as input, identifies a target object within the image, and then searches for similar objects in a specified video.

Features:

Object Detection and Tracking: Uses YOLOv8 for robust object detection and tracking in video streams.

Feature Extraction: Employs a fine-tuned ResNet50 model to extract features from detected objects.

Similarity Comparison: Calculates cosine similarity between the target object's features and features of objects detected in the video.

Visualization: Displays bounding boxes and similarity scores around detected objects in real-time on the video feed.

Output Generation: Allows downloading videos of the top-k most similar objects, including annotations.

Database Integration: Stores object information, features, and bounding boxes in a SQLite database for persistence and efficient data management.

Installation:

Clone the repository: git clone https://github.com/Ikromov247/forensicsUI.git

Navigate to the project directory: cd forensicsUI

Create a virtual environment (recommended): python3 -m venv .venv

Activate the virtual environment: source .venv/bin/activate

Install the required packages: pip install -r requirements.txt

Usage:

Provide Input: Place your target image (target_image.jpg) and input video (input_video.mp4) in the files directory.

Run the Application: Execute the main.py script: python main.py

Interact:

Toggle visualization on/off by pressing the 'v' key.

Press 'q' to quit the application.

Configuration:

Modify the MIN_DETECTION_CONFIDENCE variable in main.py to adjust the confidence threshold for object detection.

Adjust the extraction_interval variable to control the frequency of feature extraction in performance mode.

Update the database_name variable to specify the desired database name.

Modify paths to model files (models/YOLOv8L.pt, models/resnet50_state.pth) if necessary.

Output:

A new SQLite database (with the specified name) will be created in the database directory, containing information about detected objects and their features.

Videos of the top-k similar objects will be saved in the downloads directory, annotated with bounding boxes.

Future Enhancements:

Implement a Streamlit user interface for more interactive control and data visualization.

Add support for more sophisticated similarity metrics and feature extraction methods.

Incorporate advanced object tracking algorithms to handle occlusions and reappearances more effectively.

Integrate with cloud storage solutions for managing large datasets.

Contributing:

Contributions to the project are welcome! If you find any bugs, have suggestions for improvement, or would like to add new features, please feel free to create issues or pull requests on GitHub.

