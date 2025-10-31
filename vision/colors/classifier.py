import cv2
import numpy as np

""" Classifies face colors from frame and corners. """
class ColorClassifier:
    
    """ Sample face colors based on corner coordinates in the frame. """
    def sample_face_colors(self, frame, corners):
        # Order the corners to get a consistent perspective transform.
        ordered = self._order_points(corners)
        # Perspective transform the image into a 300x300 square.
        dst = np.array([[0, 0], [299, 0], [299, 299], [0, 299]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(ordered.astype(np.float32), dst)
        warped = cv2.warpPerspective(frame, matrix, (300, 300))

        letters = [] # Store colors of the face.
        # Parse the 3x3 grid and sample the center of each grid cell (smaller piece of the face).
        for row in range(3):
            for col in range(3):
                cx, cy = col * 100 + 50, row * 100 + 50 # Center of the grid cell.
                # Uses the median color of a smaller region in the grid cell to represent the color of the grid cell.
                region = warped[cy - 20:cy + 20, cx - 20:cx + 20]
                median_bgr = np.median(region.reshape(-1, 3), axis=0)

                # Convert BGR to HSV (Hue, Saturation, Value) for easier color classification
                hsv = cv2.cvtColor(np.uint8([[median_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
                letters.append(self._classify_hsv(hsv))
        return letters

    """ Draws a 3x3 grid preview of the classified colors on the given frame. """
    def draw_grid_preview(self, frame, letters, origin=(10, 50), size=25):
        color_map = {
            'W': (255, 255, 255), 'Y': (0, 255, 255), 'R': (0, 0, 255),
            'O': (0, 165, 255), 'B': (255, 0, 0), 'G': (0, 255, 0), '?': (128, 128, 128)
        }
        start_x, start_y = origin
        for i, letter in enumerate(letters):
            r, c = i // 3, i % 3
            x, y = start_x + c * size, start_y + r * size
            bgr = color_map.get(letter, (128, 128, 128))

            # Draw colored squares with color label
            cv2.rectangle(frame, (x, y), (x + size, y + size), bgr, -1)
            cv2.rectangle(frame, (x, y), (x + size, y + size), (0, 0, 0), 1)
            cv2.putText(frame, letter, (x + 6, y + int(size * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    """ Color classification based on HSV values. """
    def _classify_hsv(self, hsv):
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        if v > 160 and s < 65:
            return 'W'
        if v > 130 and s > 60 and 18 <= h <= 38:
            return 'Y'
        if s > 60 and (h <= 15 or h >= 165):
            return 'R'
        if s > 60 and 12 < h < 25:
            return 'O'
        if s > 60 and 80 <= h <= 140:
            return 'B'
        if s > 60 and 32 < h < 88:
            return 'G'
        return '?'

    """ Normalize corner ordering: [top-left, top-right, bottom-right, bottom-left]. """
    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-left
        rect[2] = pts[np.argmax(s)] # Bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-right
        rect[3] = pts[np.argmax(diff)] # Bottom-left
        return rect