import csv
import os

# Paths to your uploaded files (located in the same folder as this module)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OLL_PATH = os.path.join(BASE_DIR, "OLLalgos.csv")
PLL_PATH = os.path.join(BASE_DIR, "PLLalgos.csv")

def loadCSVMapping(path):
    """
    Loads a CSV and returns a dictionary:
        mapping_number -> algorithm
    The CSV is expected to contain exactly two useful columns:
        - mapping (21-bit string)
        - algorithm (text)
    Column order does not matter.
    """
    mapping = {}

    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file: {path}")

    with open(path, "r", newline='', encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # first column is the mapping key, remaining columns form the algorithm
            key = row[0].strip()
            if key == "":
                continue
            algo = ",".join(col.strip() for col in row[1:]) if len(row) > 1 else ""
            mapping[key] = algo

    return mapping


def buildCombinedMapping():
    """
    Reads both OLL and PLL CSVs, merges their mappings,
    and returns a single dictionary for full lookup.
    """
    oll = loadCSVMapping(OLL_PATH)
    pll = loadCSVMapping(PLL_PATH)

    combined = {}
    combined.update(oll)
    combined.update(pll)

    return combined