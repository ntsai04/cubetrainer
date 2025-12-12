# Cube Trainer

*A Computer Vision Project for CS 566*

**Repository:** [github.com/ntsai04/cubetrainer](https://github.com/ntsai04/cubetrainer/tree/main)

**Demo Video:** [Cube Trainer Demo](https://youtu.be/JU1_JgLlKKM)

[Download Here](../cubetrainer.exe.zip)

---

## Table of Contents
- Motivation
- Approach
- Implementation
- Results
- Discussion
- Future Work
- References

---

## Motivation

Solving a Rubik's Cube efficiently requires memorizing numerous algorithms and recognizing patterns quickly. This project aims to develop a program that can:

1. **Detect and recognize** a Rubik's Cube configuration
2. **Classify colors** and **identify patterns** corresponding to specific solving stages (OLL or PLL)
4. **Provide guidance** by providing the appropriate algorithm

The system will serve as a learning tool for speedcubing enthusiasts, helping them improve their solving techniques by reducing friction in the learning process.

### Personal Motivation
I (Nathan) have been cubing on and off for around 9 years. I began around 7th grade, and in the following years that I consistently practiced, I was able to get my best averages (ao5 and ao12) down to around 25 seconds. Over the years I came back to it occasionally, with the intent to learn full CFOP (memorizing all of the OLL and PLL algorithms), but I always fell short and it would take a while until I tried again. More recently, I ended up memorizing all of PLL (21 algorithms), around 1/3 of OLL (57 algorithms total), and have gotten my averages sub-20. I'm currently at a standstill trying to memorize the rest of the algorithms, as the remaining patterns are more difficult to distinguish. I was curious if a tool like this one would help motivate me by removing any friction in the learning process. Looking at how far we've come, I believe that with a few changes to make the program more robust (that will be mentioned later), I can use the program to help me accomplish the goal that I've been chasing for so long.

### Critical Design Specifications
- Accurate cube state detection from live video
- Robust user interface/interaction
- Real-time pattern recognition and algorithm matching for last layer algorithms

---

## Approach

Our solution combines classical computer vision techniques with a logic and algorithm hub to create a practical cube-solving assistant.

### System Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT: Camera Frame                         │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Face Detection    │
                    │   (ShapeDetector)    │
                    └──────────┬───────────┘
                               │ Corners detected
                               ▼
                    ┌──────────────────────┐
                    │ Color Classification │
                    │    (Classifier)      │
                    └──────────┬───────────┘
                               │ 3×3 color grid
                               ▼
          ┌─────────────────────────────────────────┐
          │           Virtual Cube State            │
          │  • Collect 5 faces                      │
          │  • Normalize orientation                │
          │  • Generate 21-char state string        │
          └───────────────────┬─────────────────────┘
                              │ State string
                              ▼
                   ┌──────────────────────┐
                   │   Pattern Matching   │
                   │  (AlgorithmMatcher)  │
                   └──────────┬───────────┘
                              │ Algorithm found
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│             OUTPUT: Algorithm + Pattern Label                   │
└─────────────────────────────────────────────────────────────────┘
```

### System Overview

The cube trainer operates through a five-stage pipeline that transforms camera input into algorithm recommendations:

#### 1. Face Detection
The system continuously analyzes video frames to locate cube faces. Using edge detection and contour analysis, it identifies square regions in the frame and validates them geometrically to ensure they represent actual cube faces. Once a valid face is detected, the corner positions are determined with sub-pixel precision.

#### 2. Color Classification
With the face corners identified, the system applies a perspective transformation to normalize the viewing angle, creating a uniform grid. Each of the 9 stickers is sampled by their median color, and colors are classified using HSV thresholds.

#### 3. Virtual Cube State Construction
As the user captures each face, the system builds a virtual cube representation. Colors are encoded as integers (Y=1, G=2, O=3, B=4, R=5) for efficient processing. After 5 faces are captured, the cube automatically normalizes its orientation by sorting faces by their center colors and rotating each face until edge stickers align properly.

#### 4. Pattern Matching
The oriented cube generates a 21-character state string that encodes the last layer configuration. The system first determines whether this is an OLL case (orientation needed) or PLL case (permutation needed) by checking if the top face is fully yellow. It then searches through a combined database of 78 patterns (57 OLL + 21 PLL), trying all four U-face rotations to find a match.

#### 5. Algorithm Output
Once a pattern match is found, the system returns the corresponding algorithm notation along with any necessary setup moves (U, U2, or U'). The pattern name is also displayed to help users learn to recognize these configurations visually.

---

## Technical Implementation

This section details the code-level implementation of each pipeline stage, including key algorithms, data structures, and OpenCV operations.

### Program Structure

```
cubetrainer/
├── backend/
│   ├── AlgorithmMatcher.py    # Pattern matching logic
│   ├── Cube.py                 # Cube state representation
│   ├── Face.py                 # Individual face handling
│   ├── Patterns.py             # Pattern definitions
│   ├── OLLalgos.csv           # OLL algorithm database
│   └── PLLalgos.csv           # PLL algorithm database
├── vision/
│   ├── colors/
│   │   └── Classifier.py      # Color detection & classification
│   └── detection/
│       └── ShapeDetector.py   # Cube face detection
├── run.py                      # Main application entry point
```

### Vision Components

#### 1. Face Detection (`ShapeDetector.py`)

```python
def detectFace(self, frame):
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.dilate(edges, kernel, iterations=2)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, eps * perimeter, True)
        if len(approx) == 4 and cv2.contourArea(approx) > min_area:
            if self._isSquare(approx) and self._hasRightAngles(corners):
                cv2.cornerSubPix(gray_full, corners, (5, 5), (-1, -1), criteria)
                return corners

def _isSquare(self, corners):
    rect = cv2.minAreaRect(corners)
    w, h = rect[1]
    return max(w, h) / min(w, h) < 1.8

def _hasRightAngles(self, corners):
    for i in range(4):
        v1 = corners[i] - corners[(i + 1) % 4]
        v2 = corners[(i + 2) % 4] - corners[(i + 1) % 4]
        angle = np.degrees(np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
        if angle < 55 or angle > 125:
            return False
    return True
```

**Edge Detection:** Canny thresholds (40, 120) detect edges while minimizing noise. Dilation with a 3×3 kernel fills gaps from cube face imperfections.  
**Contour Filtering:** Douglas-Peucker algorithm (`cv2.approxPolyDP`) simplifies contours with epsilon based on perimeter length. Area threshold (`min_area = max(3000 * scale², 0.01 * frame_area)`) filters out small stickers.  
**Geometric Validation:** `cv2.minAreaRect` computes the minimum bounding rectangle for aspect ratio checking. Corner angles use vector dot products and arccos. Sub-pixel refinement uses a 5×5 search window with EPS+MAX_ITER termination criteria.

---

#### 2. Color Classification (`Classifier.py`)

```python
def sampleFaceColors(self, frame, corners):
    ordered = self._orderPoints(corners)
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(frame, matrix, (300, 300))
    
    letters = []
    for row in range(3):
        for col in range(3):
            cx, cy = col * 100 + 50, row * 100 + 50
            region = warped[cy - 20:cy + 20, cx - 20:cx + 20]
            median_bgr = np.median(region.reshape(-1, 3), axis=0)
            hsv = cv2.cvtColor(np.uint8([[median_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            letters.append(self._classifyHSV(hsv))
    return letters

def _classifyHSV(self, hsv):
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if s > 40 and v > 100 and 15 <= h <= 40: return 'Y'
    if s > 40 and v > 80 and (h <= 10 or h >= 170): return 'R'
    if s > 40 and v > 60 and 80 <= h <= 140: return 'B'
    if s > 40 and v > 80 and 8 < h < 20: return 'O'
    if s > 40 and v > 60 and 32 < h < 88: return 'G'
    return 'W'
```

**Perspective Transform:** 3×3 homography matrix maps arbitrary quadrilateral to 300×300 square destination. Corner ordering (top-left, top-right, bottom-right, bottom-left) determined by coordinate sum/difference.  
**Sampling Strategy:** 40×40 pixel regions (1600 samples per sticker) centered at each 100×100 grid cell. Median aggregation (`np.median`) provides robustness against specular highlights and shadows.  
**HSV Classification:** OpenCV's BGR2HSV uses H ∈ [0, 180], S ∈ [0, 255], V ∈ [0, 255]. Hue thresholds are empirically tuned for standard cube colors. Red's disjoint range (h ≤ 10 OR h ≥ 170) handles cylindrical hue space wraparound.

---

### State Management Components

#### 1. Virtual Cube (`Cube.py` + `Face.py`)

```python
class Face:
    def setFace(self, face):
        mapping = {'Y': 1, 'G': 2, 'O': 3, 'B': 4, 'R': 5}
        self.colors = [mapping.get(str(c).upper(), -1) for c in face]

class Cube:
    def addFace(self, face):
        self.faces.append(face)
        if len(self.faces) == 5:
            self.orientCube()
    
    def getCube(self):
        out = ''.join(str(int(v)) for v in self.faces[0].colors)
        for i in range(1, 5):
            out += ''.join(str(int(v)) for v in self.faces[i].colors[:3])
        return out
    
    def orientCube(self):
        orientedCube = [None] * 5
        for face in self.faces:
            orientedCube[face.colors[4] - 1] = face
        for i in range(1, 5):
            while not all(ch == face.colors[-6:][0] for ch in face.colors[-6:]):
                orientedCube[i] = self.rotateFace(orientedCube[i])
    
    def rotateFace(self, face):
        original = face.colors
        return [original[6], original[3], original[0],
                original[7], original[4], original[1],
                original[8], original[5], original[2]]
```

**Data Structure:** Integer encoding reduces memory footprint and enables fast string concatenation. Center color (index 4) serves as face identifier.  
**State String Encoding:** 21-character representation optimized for last-layer solving: full top face (9 chars) + first 3 stickers from each of 4 sides (12 chars). Omits bottom face and non-visible side stickers.  
**Orientation Algorithm:** Sorts faces into array by center color ID. Rotation invariant: checks if last 6 stickers are homogeneous (indicates proper alignment with adjacent faces).  
**Rotation Transform:** 90° clockwise mapping follows pattern: `corners → [6,0,2,8]` and `edges → [3,1,5,7]` in new positions. Equivalent to transpose + reverse rows.

---

#### 2. Algorithm Matching (`AlgorithmMatcher.py` + `Patterns.py`)

```python
def buildCombinedMapping():
    oll = loadCSVMapping(OLL_PATH)
    pll = loadCSVMapping(PLL_PATH)
    combined = {}
    combined.update(oll)
    combined.update(pll)
    return combined

class AlgorithmMatcher:
    def match(self, cube):
        cubeString = cube.getCube()
        yellowFace = cubeString[:9]
        
        if any(c != '1' for c in yellowFace):
            cubeString = ''.join(c if c == '1' else '0' for c in cubeString)
        
        attempts = 0
        while attempts < 4:
            algo = PATTERN_MAP.get(cubeString)
            if algo:
                prefix = ['', 'U ', "U2 ", "U' "][attempts]
                return prefix + algo
            attempts += 1
            cubeString = self.rotate(cubeString)
        
        return "No matching algorithm found."
    
    def rotate(self, cubeString):
        yellowFace = cubeString[:9]
        rotatedYellow = (yellowFace[6] + yellowFace[3] + yellowFace[0] + 
                         yellowFace[7] + yellowFace[4] + yellowFace[1] + 
                         yellowFace[8] + yellowFace[5] + yellowFace[2])
        rest = cubeString[9:]
        return rotatedYellow + rest[3:] + rest[:3]
```

**Database Structure:** Hash map (O(1) lookup) with 21-char keys. CSV parsing with `csv.reader` loads patterns at startup. Dictionary merge prioritizes PLL entries if collision occurs.  
**Pattern Discrimination:** Binary conversion for OLL: all non-yellow stickers → '0', preserves only orientation information. PLL preserves full color encoding (1-5) to distinguish permutation patterns.  
**Rotation-Invariant Search:** Exhaustive 4-iteration search (0°, 90°, 180°, 270°) with early termination. Each rotation applies U-turn transform to state string. Prefix notation (U, U2, U') instructs user on setup moves before executing algorithm.  
**U-Turn Simulation:** Top face rotation reuses `rotateFace` mapping. Side sticker cycling: 12-char sequence shifts left by 3 positions (equivalent to circular array rotation).

---

### Pattern Databases

The pattern databases provide the lookup tables for the algorithm matcher. Each CSV file maps state strings to solving algorithms.

#### OLL Algorithms (`OLLalgos.csv`)

Contains 57 orientation patterns. Each row maps a 21-character binary pattern to an algorithm.

**Sample Entries:**

```csv
000010000010111010111,R U2 R2 F R F' U2 R' F R F' (Dot)
110110000011011000001,I' U2 L U L' U I (Square Shape)
010110100011011001000,r U R' U R U2 r' (Small Lightning Bolt)
010110001110010100100,R U R' U' R' F R2 U R' U' F' (Fish Shape)
010111010101000101000,R U2 R' U' R U R' U' R U' R' (Cross)
```

**Format:**
- **Column 1:** 21-character pattern string (9 for top face + 3×4 for side faces)
- **Column 2:** Algorithm notation with descriptive label

---

#### PLL Algorithms (`PLLalgos.csv`)

Contains 21 permutation patterns. Since orientation is already correct, patterns use only the center color reference (1 for yellow).

**Sample Entries:**

```csv
111111111344553425232,x L2 D2 L' U' L D2 L' U L' (Aa Perm)
111111111542355223434,x' L2 D2 L U L' D2 L U' L (Ab Perm)
111111111345254523432,x' L' U L D' L' U' L D L' U' L D' L' U L D (E Perm)
111111111425254542333,R' U' F' R U R' U' R' F R2 U' R' U' R U R' U R (F Perm)
111111111455234542323,R2 U R' U R' U' R U' R2 U' D R' U R D' (Ga Perm)
```

**Format:**
- **Column 1:** 21-character pattern (all 1s in top face since it's oriented)
- **Column 2:** Algorithm notation with permutation name

---

## Challenges Encountered

#### 1. **Lighting Sensitivity & Edge Detection**
**Problem:** Color detection becomes unreliable under strong reflections on cube stickers. For example, reflections from a bright white light bulb can cause yellow stickers to be misclassified as white. In addition, complex background textures—such as wood grain patterns—can interfere with contour extraction, leading to failures in cube edge detection.

![White vs Yellow Confusion](webpage%20images/whiteyellow.png)

**Approach:** 
Although this issue cannot be fully eliminated under uncontrolled lighting, several optimizations were added to improve robustness:

- **HSV-based classification** reduced sensitivity to brightness variations compared to RGB
- **Median color sampling** within each sticker region reduced noise from specular highlights
- **Contour filtering by area and shape** helped reject background patterns that were mistakenly detected as potential cube faces
- **Perspective normalization** after detection stabilized the color sampling process, making classification more consistent across frames

**Result:** These optimizations significantly improved color and edge detection under typical indoor lighting, but strong reflections and busy backgrounds still remain challenging. The system now performs reliably when used against a simple backdrop with stable ambient lighting, but highly reflective environments may still cause classification errors.

#### 2. **Cube Orientation Detection**
**Problem:** The algorithm database specifies strict cube orientations for every OLL/PLL case. If a user captures images in a different orientation, the resulting 21-bit state will not match any entry in the database, causing no algorithm to be found. Asking users to pre-align the cube correctly is impractical and defeats the purpose of an intuitive solving assistant.

**Approach:**
- Implemented a rotation method that systematically rotates the interpreted cube state whenever no direct match is found
- Re-check the database after each rotation attempt
- Once a match is found, apply the inverse rotation to the output algorithm so the solution aligns with the cube orientation the user is actually holding

![Misoriented Cube Detection](webpage%20images/misoriented.png)

**Result:** The system can now correctly identify the intended OLL/PLL case regardless of input orientation, ensuring consistent and accurate algorithm retrieval without requiring users to manually orient the cube beforehand.



## Results + Discussion

### Detection Performance
Under controlled conditions, a clear, uniform background (black or dark tabletop) and even ambient lighting with no shadows on the cube—our system performs with near-perfect reliability:
- **Cube face detection:** The contour-based shape detector identifies cube faces consistently and accurately. In controlled tests, no false detections or missed detections occurred.
- **Color recognition:** With stable lighting and no glare, the HSV classifier correctly classified all 9 stickers per face with almost zero error across multiple test images.


### Pattern Matching Accuracy

Once the five required faces of the cube are captured:

- The system successfully constructs the 21-bit state representation.
- It then traverses the OLL/PLL algorithm database.
- Every detected state **(100% of cases)** matched the correct algorithm in our tests.

This includes:

- Correct identification of OLL orientation patterns
- Correct recognition of PLL permutation cases
- Accurate handling of cube rotations before matching

### End-to-End System Reliability

When operated under recommended conditions, the pipeline—from image capture → face extraction → color classification → state mapping → pattern retrieval—achieves full correctness for both recognition and algorithm lookup.

These results validate the robustness of our pipeline for real-time cube-state analysis, especially when used as a teaching or training tool.

---

## Comparisons
To evaluate the effectiveness of our Cube Trainer system, we compared it against an existing online Rubik’s Cube solver. The comparison focuses on four key components: **color detection, edge detection, and user interface**.

### Color Detection
Our program uses HSV-based color classification with median sampling across each sticker region. Under controlled lighting, it achieves near-perfect accuracy, though glare or reflections can still cause occasional misclassification.

The online solver relies on manual calibration before sampling from a static image. Its accuracy depends heavily on uniform lighting and often struggles with similar issues such as reflections or low contrast.

#### Example Images
![Successful Color Detection](webpage%20images/successful.png)
![Correct Detection](webpage%20images/correct.png)

### Edge Detection
Our system relies on contour-based detection to locate the cube face. While this works well on uniform or dark backgrounds, it struggles under complex background textures. For example, wood grain patterns on a tabletop often produce strong edges that get mistaken for cube boundaries, leading to false detections or unstable face tracking.

The online solver experiences similar limitations. Its detection pipeline also cannot reliably separate cube edges from textured or cluttered backgrounds. To avoid this issue, the online solver requires users to place the cube inside a predefined on-screen box, ensuring the cube is isolated and centered during image capture.

#### Example Images:
![Wood Table Background Issue](webpage%20images/wood%20table.png)
![Misoriented Detection](webpage%20images/misoriented.png)

### User Interface
Our system allows users to capture cube faces without requiring any specific orientation or order. The program automatically rotates, aligns, and stitches the captured faces internally, enabling a more natural and flexible interaction. Users can focus on simply showing each face to the camera without following strict instructions.

In contrast, the online solver enforces a step-by-step capture process, directing users to rotate the cube in specific ways and align it precisely as instructed. This ensures correct input for the solver but increases the cognitive and interaction burden on the user.

#### Example Images

![Confusing User Input Interface](webpage%20images/confusinguserinput.png)

---

## Future Work
- Make edge detection more robust -> able to work both while holding the cube and against complex backgrounds
- Implement color neutrality -> more advanced pattern recognition used by speedcubers for solve efficiency
- Automate scanning process by capturing faces without user input -> speed up overall algorithm delivery
- Improve color classification methods to reduce error while scanning
- Potentially implement deep learning/AI model to recognize cube in various orientations -> improve detection in dynamic lighting/environments

---

## References

### Academic Papers
[1] T. N. Suharsono, A. Rozak, and R. Mardiati, “Rubik's Cube Solution Method Using Real-Time Tracking Cube Approach,” in Proc. 2022 8th International Conference on Wireless and Telematics (ICWT), 2022, pp. 1–6. doi: 10.1109/ICWT55831.2022.9935436.

### Online Resources
- [OpenCV Documentation](https://docs.opencv.org/)
- [JPerm](https://jperm.net/) - Rubik's Cube Algorithms
- [Online Solver](https://rubiks-cube-solver.com/scan/) - Rubik's Cube Solver
- [Alternative Implementation](https://programmablebrick.blogspot.com/2017/02/rubiks-cube-tracker-using-opencv.html) - Preprocessing example

## Team
**Authors:** Nathan Tsai, Yibing Shen

**Course:** CS 566 - Computer Vision | Fall 2025

---

*Last Updated: December 2025*