import cv2

from vision.detection.ShapeDetector import ShapeDetector
from vision.colors.Classifier import ColorClassifier
from backend.Cube import Cube
from backend.Face import Face
from backend.AlgorithmMatcher import AlgorithmMatcher

def main():
    detector = ShapeDetector(debug=False)
    classifier = ColorClassifier()

    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit.")
    print("Press spacebar to capture the currently detected face colors.")
    print("Press 'r' to reset the captured cube faces.")

    letters = None
    cube = Cube()
    matcher = AlgorithmMatcher()
    algorithm = None
    status_msg = None
    status_frames = 0
    persistent_msgs = []

    # Main loop for live-updating UI and detection/color classification.
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy() # Use copy for UI display and keep frame for processing.

        # Detect the face and get the corners.
        corners = detector.detectFace(frame)
        if corners is not None:
            # Draw outlines on display UI.
            cv2.polylines(display, [corners], True, (0, 255, 0), 3) # Wrapped corners in green.
            # Draw bounding rectangle on display UI (wrapped outline is inscribed in this bounding rectangle).
            x, y, w, h = cv2.boundingRect(corners)
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Get the face colors and draw the grid preview on the display.
            letters = classifier.sampleFaceColors(frame, corners)
            classifier.drawGridPreview(display, letters)

        # Build a single list of messages to display so they don't overlap.
        # Start with persistent messages (captured faces), then dynamic ones.

        # If a full cube (5 faces) has been captured, attempt to compute algorithm.
        if cube.faceCount() == 5:
            if algorithm is None:
                algorithm = matcher.match(cube)

        # Compose display messages in a consistent order to avoid overlap.
        display_msgs = []
        # persistent messages first
        display_msgs.extend(persistent_msgs)
        # dynamic messages next
        if algorithm is not None:
            display_msgs.append(f"Algorithm: {algorithm}")
        # transient status (only while status_frames > 0)
        if status_frames > 0 and status_msg:
            display_msgs.append(status_msg)
            status_frames -= 1

        # draw all messages stacked vertically, starting at y=360
        for idx, msg in enumerate(display_msgs):
            y = 360 + idx * 20
            cv2.putText(display, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("Cube Trainer", display)
        
        key = cv2.waitKey(1) & 0xFF

        # Spacebar (32) to capture and save current detected face colors to backend.
        if key == 32:  # space
            if letters is None:
                status_msg = "No face detected to capture."
                status_frames = 90
            else:
                face = Face(letters)
                if face.getFace() is None:
                    status_msg = "Face contains unknown colors, cannot add."
                    status_frames = 90
                    continue
                if cube.addFace(face) is False:
                    status_msg = "Cube already has 5 faces, cannot add more."
                    status_frames = 90
                    continue
                persistent_msgs.append(f"Captured the {letters[4]} face")

        if key == ord('q'):
            break

        if key == ord('r'):
            cube = Cube()
            algorithm = None
            letters = None
            status_msg = "Reset captured cube faces."
            status_frames = 90
            # clear persistent captured messages when resetting
            persistent_msgs = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()