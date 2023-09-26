from pydantic import BaseModel

MUSHROOM_CHARACTERISTICS = {
    "cap_shape": {
        "bell": "b",
        "conical": "c",
        "convex": "x",
        "flat": "f",
        "knobbed": "k",
        "sunken": "s",
    },
    "cap_surface": {"fibrous": "f", "grooves": "g", "scaly": "y", "smooth": "s"},
    "cap_color": {
        "brown": "n",
        "buff": "b",
        "cinnamon": "c",
        "gray": "g",
        "green": "r",
        "pink": "p",
        "purple": "u",
        "red": "e",
        "white": "w",
        "yellow": "y",
    },
    "bruises": {"bruises": "t", "no": "f"},
    "odor": {
        "almond": "a",
        "anise": "l",
        "creosote": "c",
        "fishy": "y",
        "foul": "f",
        "musty": "m",
        "none": "n",
        "pungent": "p",
        "spicy": "s",
    },
    "gill_attachment": {
        "attached": "a",
        "descending": "d",
        "free": "f",
        "notched": "n",
    },
    "gill_spacing": {"close": "c", "crowded": "w", "distant": "d"},
    "gill_size": {"broad": "b", "narrow": "n"},
    "gill_color": {
        "black": "k",
        "brown": "n",
        "buff": "b",
        "chocolate": "h",
        "gray": "g",
        "green": "r",
        "orange": "o",
        "pink": "p",
        "purple": "u",
        "red": "e",
        "white": "w",
        "yellow": "y",
    },
    "stalk_shape": {"enlarging": "e", "tapering": "t"},
    "stalk_root": {
        "bulbous": "b",
        "club": "c",
        "cup": "u",
        "equal": "e",
        "rhizomorphs": "z",
        "rooted": "r",
        "missing": "?",
    },
    "stalk_surface_above_ring": {
        "fibrous": "f",
        "scaly": "y",
        "silky": "k",
        "smooth": "s",
    },
    "stalk_surface_below_ring": {
        "fibrous": "f",
        "scaly": "y",
        "silky": "k",
        "smooth": "s",
    },
    "stalk_color_above_ring": {
        "brown": "n",
        "buff": "b",
        "cinnamon": "c",
        "gray": "g",
        "orange": "o",
        "pink": "p",
        "red": "e",
        "white": "w",
        "yellow": "y",
    },
    "stalk_color_below_ring": {
        "brown": "n",
        "buff": "b",
        "cinnamon": "c",
        "gray": "g",
        "orange": "o",
        "pink": "p",
        "red": "e",
        "white": "w",
        "yellow": "y",
    },
    "veil_type": {"partial": "p", "universal": "u"},
    "veil_color": {"brown": "n", "orange": "o", "white": "w", "yellow": "y"},
    "ring_number": {"none": "n", "one": "o", "two": "t"},
    "ring_type": {
        "cobwebby": "c",
        "evanescent": "e",
        "flaring": "f",
        "large": "l",
        "none": "n",
        "pendant": "p",
        "sheathing": "s",
        "zone": "z",
    },
    "spore_print_color": {
        "black": "k",
        "brown": "n",
        "buff": "b",
        "chocolate": "h",
        "green": "r",
        "orange": "o",
        "purple": "u",
        "white": "w",
        "yellow": "y",
    },
    "population": {
        "abundant": "a",
        "clustered": "c",
        "numerous": "n",
        "scattered": "s",
        "several": "v",
        "solitary": "y",
    },
    "habitat": {
        "grasses": "g",
        "leaves": "l",
        "meadows": "m",
        "paths": "p",
        "urban": "u",
        "waste": "w",
        "woods": "d",
    },
}


class Mushroom(BaseModel):
    cap_shape: str
    cap_surface: str
    cap_color: str
    bruises: str
    odor: str
    gill_attachment: str
    gill_spacing: str
    gill_size: str
    gill_color: str
    stalk_shape: str
    stalk_root: str
    stalk_surface_above_ring: str
    stalk_surface_below_ring: str
    stalk_color_above_ring: str
    stalk_color_below_ring: str
    veil_type: str
    veil_color: str
    ring_number: str
    ring_type: str
    spore_print_color: str
    population: str
    habitat: str
