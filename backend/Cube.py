"""Simple Cube model to hold captured faces."""
from .Face import Face

class Cube:
    """A minimal container for captured Face instances."""
    def __init__(self):
        self.faces = []

    def addFace(self, face: Face) -> bool:
        if (len(self.faces) == 5):
            return False
        self.faces.append(face)
        if len(self.faces) == 5:
            self.orientCube()
        return True

    def faceCount(self) -> int:
        return len(self.faces)

    def getCube(self) -> str:
        # Return a string that is the concatenation of each face's
        # integer representation (each face is 9 digits long).
        if len(self.faces) != 5:
            return None

        out = ''
        for i, face in enumerate(self.faces):
            # assume face.colors is a list of integers (length 9)
            face_str = ''.join(str(int(v)) for v in face.colors)
            if i == 0:
                out += face_str
            else:
                # append only the first 3 digits of remaining faces
                out += face_str[:3]

        return out
    
    def orientCube(self):
        orientedCube = [None] * 5

        for face in self.faces:
            orientedCube[face.colors[4] - 1] = face

        # Rotate faces from the second face (index 1) to the last one.
        # The first face (index 0) is assumed to be in correct rotation.
        for i in range(1, 5):
            face = orientedCube[i]

            while True:
                face_str = ''.join(str(int(v)) for v in face.colors)
                last6 = face_str[-6:]
                if all(ch == last6[0] for ch in last6):
                    break
                face = self.rotateFace(face)

            orientedCube[i] = face

        self.faces = orientedCube

    def rotateFace(self, face: Face) -> Face:
        """Rotate a face 90 degrees clockwise."""
        original = face.colors
        rotated = [
            original[6], original[3], original[0],
            original[7], original[4], original[1],
            original[8], original[5], original[2],
        ]
        # Mutate the provided Face instance to preserve integer colors
        # and avoid calling Face.__init__ which expects letter tokens.
        face.colors = rotated
        return face

    def __repr__(self):
        return f"Cube(faces={self.faces})"
