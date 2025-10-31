import cv2

from vision.detection.shape_detector import ShapeDetector
from vision.colors.classifier import ColorClassifier

def main():
    detector = ShapeDetector(debug=True)
    classifier = ColorClassifier()

    cap = cv2.VideoCapture(1)

    print("Press 'q' to quit.\n")

    # Track runtime stats for performance monitoring.
    no_detect_streak = 0
    frame_count = 0

    # Main loop for live-updating UI and detection/color classification.
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display = frame.copy() # Use copy for UI display and keep frame for processing.

        # Show no face detected message if no face is detected for 10 frames.
        if no_detect_streak >= 10 and frame_count % 2 == 1:
            cv2.putText(display, "No cube face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Cube Trainer", display)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
            continue

        # Detect the face and get the corners.
        corners = detector.detect_face(frame)
        if corners is not None:
            # Draw outlines on display UI.
            cv2.polylines(display, [corners], True, (0, 255, 0), 3) # Wrapped corners in green.
            # Draw bounding rectangle on display UI (wrapped outline is inscribed in this bounding rectangle).
            x, y, w, h = cv2.boundingRect(corners)
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Get the face colors and draw the grid preview on the display.
            letters = classifier.sample_face_colors(frame, corners)
            classifier.draw_grid_preview(display, letters)

            no_detect_streak = 0 # Reset the no detect streak.
        else:
            no_detect_streak += 1

        cv2.imshow("Cube Trainer", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()