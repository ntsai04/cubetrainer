# -*- coding: utf-8 -*-
import cv2
import numpy as np

""" Main detecting class for cube and faces. """
class ShapeDetector:

    """ Optional debugging UI and max width and contours for performance. """
    def __init__(self, debug=False, max_width=480, max_contours=300):
        self.debug = debug
        self.max_width = max_width
        self.max_contours = max_contours

    """ Finds corners of detected face. Returns 4 corners or None. """
    def detect_face(self, frame):
        h, w = frame.shape[:2]
        scale = min(1.0, self.max_width / w)
        # Only resize if max_width is smaller than the frame.
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame

        # Set of frame processing steps to get accurately get edges via Canny edge detection.
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 120)
        
        # Enlarge the edges and fill any small gaps (useful for imperfect Rubik's Cube faces).
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        if self.debug:
            cv2.imshow("Edges", edges)

        # Finds best contour candidates and sorts them by area to prioritize larger contours (more likely to be the entire face).
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Filter out inner contours (center, edge, corner faces) and simplify contour lines for performance.
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:self.max_contours] # Limit count to max_contours for performance.

        # Set a minimum area that the contour must meet to be considered a cube face (filter out the smaller center, edge, and corner faces).
        min_area = max(3000 * (scale ** 2), 0.01 * small.shape[0] * small.shape[1])

        # Filter contours based on geometric properties to keep most valid faces.
        candidates = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            
            for eps in (0.02, 0.03, 0.04):
                # Get a simplified polygon approximation of the contour to determine squareness.
                approx = cv2.approxPolyDP(contour, eps * perimeter, True)
                
                # Contours with 4 corners and sufficient area are passed through geometric checks next.
                if len(approx) == 4 and cv2.contourArea(approx) > min_area:
                    corners = approx.reshape(4, 2) # Format corners.
                    
                    if (self._is_square(approx) and 
                        self._has_right_angles(corners) and
                        not self._touches_edge(corners, small.shape)):
                        
                        # Remember to scale corners back to original frame.
                        if scale < 1.0:
                            corners = (corners / scale).astype(np.int32)
                        
                        score = self._score_square(corners) # Give the candidate face a squareness score.
                        candidates.append({'corners': corners, 'score': score}) # Keep track of candidates.
                        break

        if not candidates:
            return None

        best = max(candidates, key=lambda x: x['score']) # Use the highest scoring candidate.

        if self.debug:
            debug = frame.copy()
            cv2.polylines(debug, [best['corners']], True, (255, 0, 255), 3)
            cv2.imshow("Detection", debug)

        # Use cornerSubPix to refine the corners for more accurate position.
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = best['corners'].astype(np.float32).reshape(-1, 1, 2)
        cv2.cornerSubPix(gray_full, corners, (5, 5), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.03))
        return corners.reshape(-1, 2).astype(np.int32)

    """ Assign a squareness score to given face based on aspect ratio and angle variance. """
    def _score_square(self, corners):
        score = 0.0
        
        rect = cv2.minAreaRect(corners.reshape(-1, 1, 2))
        w, h = rect[1]
        if w > 0 and h > 0:
            score += (min(w, h) / max(w, h)) * 50 # Score based on closeness to square aspect ratio.
        
        angles = []
        for i in range(4):
            v1 = corners[i] - corners[(i + 1) % 4]
            v2 = corners[(i + 2) % 4] - corners[(i + 1) % 4]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            angles.append(angle)
        
        # Score based on angle variance.
        variance = np.var([abs(a - 90) for a in angles])
        score += max(0, 50 - variance)
        
        return score

    """ Determine if the face is a square. """
    def _is_square(self, corners):
        rect = cv2.minAreaRect(corners)
        w, h = rect[1]
        return w > 0 and h > 0 and max(w, h) / min(w, h) < 1.8 # Leniant square aspect ratio.

    """ Determine if the face has 90 degree angles. """
    def _has_right_angles(self, corners):
        for i in range(4):
            v1 = corners[i] - corners[(i + 1) % 4]
            v2 = corners[(i + 2) % 4] - corners[(i + 1) % 4]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.degrees(np.abs(np.arccos(np.clip(cos_angle, -1, 1))))
            if angle < 55 or angle > 125: # Leniant angle variance.
                return False
        return True

    """ Determine if the face touches the edge of the frame (prevent border issues). """
    def _touches_edge(self, corners, shape):
        h, w = shape[:2]
        margin = 5
        for x, y in corners:
            if margin < x < w - margin and margin < y < h - margin:
                return False
        return True