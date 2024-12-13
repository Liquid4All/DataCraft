from typing import Dict, List
import os
import json
from datasets import load_dataset
from utils.format import name_url
from utils.dataframe import BaseData, HFStaticDataIterator
import hashlib
import inflect

def format_points(points, label):
    return [
        {
            "label": label,
            "coordinates": [point["x"], point["y"]]
        } for point in points
    ]

def normalize_point(point, decimals=3):
    """Normalize x,y coordinates from 0-100 to 0-1 with specified decimal places"""
    return {
        "x": round(point["x"] / 100, decimals),
        "y": round(point["y"] / 100, decimals)
    }

p = inflect.engine()

def pluralize(word):
    # Check if the word is plural
    try:
        if p.singular_noun(word):
            return word
        else:
            return p.plural(word)
    except:
        return word

def singularize(word):
    # Check if the word is plural
    try:
        singular_form = p.singular_noun(word)
        if singular_form:
            return singular_form
        else:
            return word
    except:
        return word

COUNT_QUESTION_TEMPLATES = [
    lambda label: f"How many {pluralize(label)} are there?",
    lambda label: f"How many {pluralize(label)}?",
    lambda label: f"How many {pluralize(label)}.",
    lambda label: f"how many {pluralize(label)}.",
    lambda label: f"how many {pluralize(label)}?",
    lambda label: f"How many {pluralize(label)} are there in the image?",
    lambda label: f"Tell me how many {pluralize(label)} there are",
    lambda label: f"how many {pluralize(label)}",
    lambda label: f"Tell me how many {pluralize(label)} are in the image",
    lambda label: f"count {pluralize(label)}",
    lambda label: f"count every {singularize(label)}",
    lambda label: f"count each {singularize(label)}",
    lambda label: f"Count the {pluralize(label)}.",
    lambda label: f"How many {pluralize(label)} do you see?",
    lambda label: f"How many {pluralize(label)} are visible?",
    lambda label: f"Count all the {pluralize(label)}",
    lambda label: f"how many {pluralize(label)}?",
    lambda label: f"Count every {singularize(label)} in the picture.",
    lambda label: f"Count all the {pluralize(label)}",
    lambda label: f"Count each {singularize(label)}",
    lambda label: f"What is the total number of {pluralize(label)} in the image?",
    lambda label: f"In all the picture, how many {pluralize(label)} are there?",
    lambda label: f"How many {pluralize(label)} are there in the image?",
    lambda label: f"Give me the count of {pluralize(label)} in the image.",
    lambda label: f"How many {pluralize(label)} are visible in the image?",
    lambda label: f"How many {pluralize(label)} are there?",
    lambda label: f"In the image, how many {pluralize(label)} are there?",
    lambda label: f"Can you count every {singularize(label)} in the picture?",
    lambda label: f"Can you see any {pluralize(label)} in the image? How many are there?",
    lambda label: f"Are there any {pluralize(label)} in the image? How many are there?",
    lambda label: f"Object: {singularize(label)}\nInstruction: How many are there?"
]

COUNT_ANSWER_TEMPLATES = {
    "zero": [
        lambda label: f"There are no {pluralize(label)} in the image.",
        lambda label: f"I don't see any {pluralize(label)} in this image.",
        lambda label: f"The image contains no {pluralize(label)}.",
        lambda label: f"No {pluralize(label)} are present in the image.",
        lambda label: f"I cannot find any {pluralize(label)} in the image.",
        lambda label: f"This image has no {pluralize(label)}.",
        lambda label: f"Zero {pluralize(label)} found in the image.",
        lambda label: f"There aren't any {pluralize(label)} in this image.",
        lambda label: f"The image does not contain any {pluralize(label)}.",
        lambda label: f"I see no {pluralize(label)} in the image.",
        lambda label: f"{singularize(label)} count: 0",
        lambda label: f"No {pluralize(label)} detected in this image.",
        lambda label: "This isn't in the image.",
        lambda label: f"I looked but couldn't find any {pluralize(label)}.",
        lambda label: f"There are zero {pluralize(label)} in this picture."
    ],
    "one": [
        lambda label: f"There is one {singularize(label)} in the image.",
        lambda label: f"I see a single {singularize(label)} in the image.",
        lambda label: f"The image contains one {singularize(label)}.",
        lambda label: f"There is exactly one {singularize(label)} in this image.",
        lambda label: f"One {singularize(label)} is present in the image.",
        lambda label: f"I found one {singularize(label)} in the image.",
        lambda label: f"Just one {singularize(label)} appears in this image.",
        lambda label: f"A single {singularize(label)} can be seen in the image.",
        lambda label: f"The image shows one {singularize(label)}.",
        lambda label: f"{singularize(label)} count: 1",
        lambda label: f"There's one {singularize(label)} in this picture.",
        lambda label: f"I detected one {singularize(label)}.",
        lambda label: f"One {singularize(label)} is visible in this image.",
        lambda label: f"There is 1 {singularize(label)} in the image.",
        lambda label: f"The total count of {singularize(label)} is one."
    ],
    "many": [
        lambda label, count: f"There are {count} {pluralize(label)} in the image.",
        lambda label, count: f"I see {count} {pluralize(label)} in the image.",
        lambda label, count: f"The image contains {count} {pluralize(label)}.",
        lambda label, count: f"I found {count} {pluralize(label)} in this image.",
        lambda label, count: f"{count} {pluralize(label)} are present in the image.",
        lambda label, count: f"The image shows {count} {pluralize(label)}.",
        lambda label, count: f"I can see {count} {pluralize(label)} in total.",
        lambda label, count: f"There are exactly {count} {pluralize(label)} in this image.",
        lambda label, count: f"I counted {count} {pluralize(label)} in the image.",
        lambda label, count: f"{singularize(label)} count: {count}",
        lambda label, count: f"Total number of {pluralize(label)}: {count}",
        lambda label, count: f"I detected {count} {pluralize(label)}.",
        lambda label, count: f"{count} {pluralize(label)} are visible in this image.",
        lambda label, count: f"The total count of {pluralize(label)} is {count}.",
        lambda label, count: f"In this image, there are {count} {pluralize(label)}."
    ]
}

POINTING_QUESTIONS = [
    # Singular form (doesn't reveal count)
    lambda label: f"Point to the {singularize(label)} using x/y coordinates.",
    lambda label: f"What is the precise location of the {singularize(label)}?",
    lambda label: f"Show me where the {singularize(label)} is in this image.",
    lambda label: f"Give me the position of the {singularize(label)}.",
    lambda label: f"Mark the exact spot of the {singularize(label)} with coordinates.",

    # Plural form (doesn't reveal count)
    lambda label: f"Point to any {pluralize(label)} using x/y coordinates.",
    lambda label: f"List the coordinates of any {pluralize(label)} in the image.",
    lambda label: f"Show me where the {pluralize(label)} are located.",
    lambda label: f"Mark any {pluralize(label)} you can find with coordinates.",
    lambda label: f"What are the positions of the {pluralize(label)}?",

    # With explicit coordinate mention
    lambda label: f"Give the x/y coordinates for any {singularize(label)} present.",
    lambda label: f"Locate the {pluralize(label)} with their x/y values.",
    lambda label: f"Provide x/y positions for any {singularize(label)} you see.",
    
    # Position/location focused
    lambda label: f"Where can I find the {singularize(label)} (in coordinate form)?",
    lambda label: f"What are the numerical positions of any {pluralize(label)}?",
    lambda label: f"Specify the coordinates where the {pluralize(label)} appear.",
    
    # Image context explicit
    lambda label: f"In this image, where is the {singularize(label)} located?",
    lambda label: f"Find any {pluralize(label)} in this image and give their coordinates.",
    lambda label: f"What are the coordinate positions of the {pluralize(label)} shown here?",
    
    # Action-oriented
    lambda label: f"Mark the location of any {singularize(label)} with coordinates.",
    lambda label: f"Help me locate the {pluralize(label)} by providing coordinates.",
    lambda label: f"Point out any {pluralize(label)} using coordinate pairs.",
    
    # Natural queries
    lambda label: f"Can you tell me where the {singularize(label)} is using coordinates?",
    lambda label: f"Would you locate any {pluralize(label)} with precise positions?",
    lambda label: f"Please find the coordinates of the {pluralize(label)}."
]

POINTING_ANSWERS = {
    "zero": [
        lambda label: f"No {pluralize(label)} found in this image.",
        lambda label: f"I don't see any {pluralize(label)}.",
        lambda label: f"The image contains no {pluralize(label)}.",
        lambda label: f"There aren't any {pluralize(label)} present.",
        lambda label: f"I cannot locate any {pluralize(label)}.",
    ],
        
    "one": [
        lambda label, point: f"I found one {singularize(label)}, {singularize(label)}_1(x:{normalize_point(point)['x']}, y:{normalize_point(point)['y']})",
        lambda label, point: f"There's a {singularize(label)} at {singularize(label)}_1(x:{normalize_point(point)['x']}, y:{normalize_point(point)['y']})",
        lambda label, point: f"I spotted a {singularize(label)} in the image, its coordinates are {singularize(label)}_1(x:{normalize_point(point)['x']}, y:{normalize_point(point)['y']})",
        lambda label, point: f"One {singularize(label)} appears at position {singularize(label)}_1(x:{normalize_point(point)['x']}, y:{normalize_point(point)['y']})",
        lambda label, point: f"The coordinates of the {singularize(label)} I found are {singularize(label)}_1(x:{normalize_point(point)['x']}, y:{normalize_point(point)['y']})",
        lambda label, point: f"Let me give you the position of the {singularize(label)} I see:\n{singularize(label)}_1(x:{normalize_point(point)['x']}, y:{normalize_point(point)['y']})",
        lambda label, point: f"A single {singularize(label)} can be found at {singularize(label)}_1(x:{normalize_point(point)['x']}, y:{normalize_point(point)['y']})",
        lambda label, point: f"Looking at the image, I can see a {singularize(label)} at {singularize(label)}_1(x:{normalize_point(point)['x']}, y:{normalize_point(point)['y']})",
        lambda label, point: f"The {singularize(label)} appears at coordinates {singularize(label)}_1(x:{normalize_point(point)['x']}, y:{normalize_point(point)['y']})",
        lambda label, point: f"I've located the {singularize(label)} at {singularize(label)}_1(x:{normalize_point(point)['x']}, y:{normalize_point(point)['y']})",
    ],
    
    "many": [
        lambda label, points: f"I found several {pluralize(label)} in the image. " + "; ".join(
            f"{singularize(label)}_{i+1}(x:{normalize_point(p)['x']}, y:{normalize_point(p)['y']})"
            for i, p in enumerate(points)
        ),
        
        lambda label, points: f"There are multiple {pluralize(label)} at these coordinates: " + "; ".join(
            f"{singularize(label)}_{i+1}(x:{normalize_point(p)['x']}, y:{normalize_point(p)['y']})"
            for i, p in enumerate(points)
        ),

        lambda label, points: f"There are multiple {pluralize(label)}: " + "; ".join(
            f"{singularize(label)}_{i+1}(x:{normalize_point(p)['x']}, y:{normalize_point(p)['y']})"
            for i, p in enumerate(points)
        ),
        
        lambda label, points: f"Let me list out all the {pluralize(label)} I found. " + "; ".join(
            f"{singularize(label)}_{i+1}(x:{normalize_point(p)['x']}, y:{normalize_point(p)['y']})"
            for i, p in enumerate(points)
        ),
        
        lambda label, points: f"I spotted {len(points)} {pluralize(label)} in total. Here they are: " + "; ".join(
            f"{singularize(label)}_{i+1}(x:{normalize_point(p)['x']}, y:{normalize_point(p)['y']})"
            for i, p in enumerate(points)
        ),
        
        lambda label, points: f"Looking at the image, I can see multiple {pluralize(label)}:\n" + "; ".join(
            f"{singularize(label)}_{i+1}(x:{normalize_point(p)['x']}, y:{normalize_point(p)['y']})"
            for i, p in enumerate(points)
        ),
        
        lambda label, points: f"The {pluralize(label)} can be found at these positions: " + "; ".join(
            f"{singularize(label)}_{i+1}(x:{normalize_point(p)['x']}, y:{normalize_point(p)['y']})"
            for i, p in enumerate(points)
        ),
        
        lambda label, points: f"Here are all the {pluralize(label)} I've located:\n" + "; ".join(
            f"{singularize(label)}_{i+1}(x:{normalize_point(p)['x']}, y:{normalize_point(p)['y']})"
            for i, p in enumerate(points)
        ),
        
        lambda label, points: f"I've found {len(points)} {pluralize(label)} and marked their positions " + "; ".join(
            f"{singularize(label)}_{i+1}(x:{normalize_point(p)['x']}, y:{normalize_point(p)['y']})"
            for i, p in enumerate(points)
        ),
        
        lambda label, points: f"Let me give you the coordinates for each {singularize(label)} I see:\n" + "; ".join(
            f"{singularize(label)}_{i+1}(x:{normalize_point(p)['x']}, y:{normalize_point(p)['y']})"
            for i, p in enumerate(points)
        ),
        
        lambda label, points: f"There are several {pluralize(label)} in different locations: " + "; ".join(
            f"{singularize(label)}_{i+1}(x:{normalize_point(p)['x']}, y:{normalize_point(p)['y']})"
            for i, p in enumerate(points)
        ),
    ]
}

def get_deterministic_index(input_string: str, max_value: int) -> int:
    """
    Generate a deterministic index from a string using MD5 hash.
    
    Args:
        input_string: String to hash
        max_value: Maximum value to modulo by
        
    Returns:
        Integer between 0 and max_value-1
    """
    # Create MD5 hash of input string
    hash_obj = hashlib.md5(input_string.encode('utf-8'))
    # Convert first 8 bytes of hash to integer
    hash_int = int.from_bytes(hash_obj.digest()[:8], byteorder='big')
    # Return modulo to get index within range
    return hash_int % max_value

def format_question(label: str, is_counting: bool = False) -> str:
    """
    Format a deterministic question for either pointing or counting.
    
    Args:
        label: The object label to ask about
        is_counting: Whether this is a counting question (vs pointing)
    
    Returns:
        Formatted question string
    """
    templates = COUNT_QUESTION_TEMPLATES if is_counting else POINTING_QUESTIONS
    
    # Create a hash input that combines label and question type
    hash_input = f"{label}_{is_counting}_question"
    template_index = get_deterministic_index(hash_input, len(templates))
    
    question_template = templates[template_index]
    return question_template(label.lower())

def format_answer(points: List[Dict[str, float]], label: str, count: int = None) -> str:
    """
    Format a deterministic answer based on points or count.
    
    Args:
        points: List of coordinate dictionaries with x,y values
        label: The object label being discussed
        count: Optional explicit count (if None, uses len(points))
    
    Returns:
        Formatted answer string
    """
    # Determine if this is a counting answer
    is_counting = count is not None
    templates = COUNT_ANSWER_TEMPLATES if is_counting else POINTING_ANSWERS
    
    # Use count if provided, otherwise use length of points
    num_items = count if is_counting else len(points)
    label = label.lower()
    
    if num_items == 0:
        # Create hash input for zero case
        hash_input = f"{label}_zero_{is_counting}_answer"
        template_index = get_deterministic_index(hash_input, len(templates["zero"]))
        answer_template = templates["zero"][template_index]
        return answer_template(label)
    
    elif num_items == 1:
        # Create hash input for single item case
        hash_input = f"{label}_one_{is_counting}_answer"
        if not is_counting:
            # Include coordinates in hash for pointing questions
            hash_input += f"_{points[0]['x']}_{points[0]['y']}"
        template_index = get_deterministic_index(hash_input, len(templates["one"]))
        answer_template = templates["one"][template_index]
        
        if is_counting:
            return answer_template(label)
        else:
            return answer_template(label, points[0])
    
    else:
        # Create hash input for multiple items case
        hash_input = f"{label}_many_{num_items}_{is_counting}_answer"
        if not is_counting:
            # Include coordinates in hash for pointing questions
            for point in points:
                hash_input += f"_{point['x']}_{point['y']}"
        template_index = get_deterministic_index(hash_input, len(templates["many"]))
        answer_template = templates["many"][template_index]
        
        if is_counting:
            return answer_template(label, num_items)
        else:
            return answer_template(label, points)

class Data(BaseData):
    info = "PIXMO Points and Counting dataset containing image-based conversations"
    data_source = "pixmo_points"
    data_type = "conversation"
    length = 272000

    def load(self):
        self.image_folder_path = os.path.join(self.data_folder_path, "pixmo-images/images")
        return HFStaticDataIterator(load_dataset("allenai/pixmo-points", split="train"))

    def batch_process(self, batch: Dict[str, List]) -> Dict[str, List]:
        images = [os.path.join(self.image_folder_path, name_url(url)) for url in batch["image_url"]]
        for url, path in zip(batch["image_url"], images):
            self.add_file(url, path)
        
        information = []
        for img_path, points, count, label, cm in zip(images, batch["points"], batch["count"], batch["label"], batch["collection_method"]):
            info = {
                "points": format_points(points, label)
            }
            
            info["conversations"] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "value": img_path},
                        {"type": "text", "value": format_question(label.replace("  ", " "), is_counting=(count is not None))}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "value": format_answer(points, label.replace("  ", " "), count)}
                    ]
                }
            ]
                
            information.append(json.dumps(info))

        return {
            "data": information,
            "files": [json.dumps([img_path]) for img_path in images]
        }