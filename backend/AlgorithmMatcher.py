from .Patterns import build_combined_mapping

PATTERN_MAP = build_combined_mapping()

class AlgorithmMatcher():
    """Placeholder function to return a dummy algorithm string."""
    def match(self, cube) -> str:
        cubeString = cube.getCube()
        yellowFace = cubeString[:9]

        # PLL matcher
        if any(c != '1' for c in yellowFace):
            cubeString = ''.join(c if c == '1' else '0' for c in cubeString)

        attempts = 0
        while attempts < 4:
            algo = PATTERN_MAP.get(cubeString)
            if algo:
                prefix = ''
                if attempts == 0:
                    prefix = ''
                elif attempts == 1:
                    prefix = 'U '
                elif attempts == 2:
                    prefix = "U2 "
                elif attempts == 3:
                    prefix = "U' "
                return prefix + algo
            attempts += 1
            cubeString = self.rotate(cubeString)

        return "No matching algorithm found."
    
    def rotate (self, cubeString: str) -> str:
        """Rotate the cube string representation 90 degrees clockwise."""
        # Rotate the first face (first 9 chars)
        yellowFace = cubeString[:9]
        rotatedYellowFace = (
            yellowFace[6] + yellowFace[3] + yellowFace[0] +
            yellowFace[7] + yellowFace[4] + yellowFace[1] +
            yellowFace[8] + yellowFace[5] + yellowFace[2]
        )
        rest = cubeString[9:]
        rotatedRest = (
            rest[3:] + rest[:3]  # Rotate the next 12 chars (4 faces of 3 chars each)
        )

        # The rest of the cube string remains unchanged in this simplified model
        return rotatedYellowFace + rotatedRest