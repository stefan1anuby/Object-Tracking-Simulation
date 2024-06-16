import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def detect_objects(frame):
    """
    Detect circles and rectangles in the frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Find circles using Hough Transform
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=20,
        maxRadius=1000,
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int").tolist()
    else:
        circles = []

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours and approximate them to rectangles
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rectangles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:  # Ignore small contours
            continue
        approx = cv2.approxPolyDP(
            contour, 0.04 * cv2.arcLength(contour, True), True
        )
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            rectangles.append((x, y, w, h))

    return circles, rectangles


def get_dominant_color(roi):
    """
    Get the dominant color in the region of interest (ROI) using KMeans clustering.
    """
    if roi.size == 0:
        return (0, 0, 0)
    roi = roi.reshape((roi.shape[0] * roi.shape[1], 3))
    kmeans = KMeans(n_clusters=1, n_init=3)
    kmeans.fit(roi)
    dominant_color = kmeans.cluster_centers_[0]
    return tuple(map(int, dominant_color))


def color_difference(color1, color2):
    """
    Calculate the difference between two colors.
    """
    return sum(abs(c1 - c2) for c1, c2 in zip(color1, color2))


class ObjectTracker:
    def __init__(self):
        """
        Initialize the object tracker.
        """
        self.history = []  # Store history of tracked objects
        self.next_id = 0  # ID counter for new objects
        self.color_id_map = {}  # Map colors to IDs
        self.objects_last_coordonate = []  # List of last coordinates and colors

    def assign_id(self, color):
        """
        Assign a unique ID to a color if not already assigned.
        """
        if color not in self.color_id_map:
            self.color_id_map[color] = self.next_id
            self.next_id += 1
        return self.color_id_map[color]

    def update(self, circles, rectangles, frame):
        """
        Update the tracker with new circles and rectangles.
        """
        frame_objects = []

        # Handle circles
        for circle in circles:
            x, y, r = circle
            color = get_dominant_color(frame[y - r : y + r, x - r : x + r])
            found = False
            for idx, (last_x, last_y, last_color, object_id) in enumerate(
                self.objects_last_coordonate
            ):
                if (
                    color_difference(last_color, color) < 50
                    and abs(x - last_x) < 20
                    and abs(y - last_y) < 20
                ):
                    frame_objects.append([x, y, r, object_id])
                    found = True
                    self.objects_last_coordonate[idx] = (x, y, color, object_id)
                    break
            if not found:
                object_id = self.assign_id(color)
                frame_objects.append([x, y, r, object_id])
                self.objects_last_coordonate.append((x, y, color, object_id))

        # Handle rectangles
        for rect in rectangles:
            x, y, w, h = rect
            color = get_dominant_color(frame[y : y + h, x : x + w])
            found = False
            for idx, (last_x, last_y, last_color, object_id) in enumerate(
                self.objects_last_coordonate
            ):
                if (
                    color_difference(last_color, color) < 50
                    and abs(x - last_x) < 50
                    and abs(y - last_y) < 50
                ):
                    frame_objects.append([x, y, w, h, object_id])
                    found = True
                    self.objects_last_coordonate[idx] = (x, y, color, object_id)
                    break
            if not found:
                object_id = self.assign_id(color)
                frame_objects.append([x, y, w, h, object_id])
                self.objects_last_coordonate.append((x, y, color, object_id))

        self.history.append(frame_objects)

    def get_path(self):
        """
        Retrieve the paths of all tracked objects.
        """
        paths = {}
        for frame_objects in self.history:
            for obj in frame_objects:
                object_id = obj[-1]
                if object_id not in paths:
                    paths[object_id] = []
                if len(obj) == 4:  # Circle
                    x, y, r = obj[:3]
                    paths[object_id].append((x, y))
                else:  # Rectangle
                    x, y, w, h = obj[:4]
                    paths[object_id].append((x, y))
        return paths


def visualize_paths(paths):
    """
    Visualize the paths of all tracked objects.
    """
    plt.figure()
    for obj_id, path in paths.items():
        xs = [pos[0] for pos in path]
        ys = [pos[1] for pos in path]
        plt.plot(xs, ys, marker="o", label=f"ID {obj_id}")
    plt.title("Object Paths")
    plt.xlabel("Frame")
    plt.ylabel("Position")
    plt.legend()
    plt.show()


class ObjectTrackerApp:
    def __init__(self, video_path):
        """
        Initialize the application with the video path and create a tracker.
        """
        self.cap = cv2.VideoCapture(video_path)
        self.tracker = ObjectTracker()

    def run(self):
        """
        Run the object tracking application.
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            circles, rectangles = detect_objects(frame)
            self.tracker.update(circles, rectangles, frame)

            # Draw detected objects on the frame
            for obj in self.tracker.history[-1]:
                if len(obj) == 4:  # It's a circle
                    x, y, r, object_id = obj
                    cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Circle ID {object_id}",
                        (x - 10, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                else:  # It's a rectangle
                    x, y, w, h, object_id = obj
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        f"Rectangle ID {object_id}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

        # Visualize the tracked paths
        paths = self.tracker.get_path()
        visualize_paths(paths)


if __name__ == "__main__":
    app = ObjectTrackerApp("video.mp4")
    app.run()
