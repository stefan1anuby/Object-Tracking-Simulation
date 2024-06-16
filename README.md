# Object Tracking Simulation

This project simulates real-time object tracking using input from a video feed, `luxonis_task_video.mp4`. It identifies and tracks the movement of circles and rectangles of various colors across video frames, visualizing their paths over time.

## Object Detection
The object detection is performed using the detect_objects function, which identifies circles and rectangles in each frame.

### Object Detector
Function: detect_objects <br />
Description: This function processes each frame to detect circles and rectangles. <br />
Output: It returns two lists: one for detected circles and one for detected rectangles. <br />

## Object Tracker
The ObjectTracker class is responsible for tracking the detected objects across frames.

### Tracker
Class: ObjectTracker <br />
Description: This class maintains a history of tracked objects and assigns unique IDs to new objects based on their color. <br />
Methods: <br />
assign_id(color): Assigns a unique ID to a new object. <br />
update(circles, rectangles, frame): Updates the tracker with new detections. <br />
get_path(): Retrieves the paths of all tracked objects. <br />
## Running the App
The ObjectTrackerApp class encapsulates the main functionality, including video capture, object detection, tracking, and visualization.

### App
Class: ObjectTrackerApp <br />
Description: This class handles the video capture and runs the object tracking loop. <br />
Methods: <br />
__init__(video_path): Initializes the application with the video path. <br />
run(): Runs the object tracking application, displays the results, and visualizes the tracked paths. <br />

## How to run the App
Ensure that the video file luxonis_task_video.mp4 is in the same directory as your script. <br />
Run the script:
```
python main.py
```

The app will open a window displaying the video with tracked objects annotated with their IDs. Press q to quit the app. After processing the video, a plot showing the paths of tracked objects will be displayed.