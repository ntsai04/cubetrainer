"""Simple Face model to hold 3x3 face colors."""

class Face:
    def __init__(self, face):
        self.colors = None
        self.setFace(face)

    def setFace(self, face):
        # switch-like mapping implemented with a dict — store integers only
        mapping = {
            'Y': 1,
            'G': 2,
            'O': 3,
            'B': 4,
            'R': 5,
        }

        ints = []
        for color in face:
            key = str(color).upper()
            if key == '?':
                return
            ints.append(mapping.get(key, -1))

        self.colors = ints
        
    def getFace(self):
        return self.colors

    def __repr__(self):
        return f"Face({self.colors})"