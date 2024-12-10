from typing import Dict, List
import os
import re
import json
from datasets import load_dataset
from utils.format import name_url
from utils.dataframe import BaseData, HFStaticDataIterator

def format_points(points, label):
    return [
        {
            "label": label,
            "coordinates": [point["x"], point["y"]]
        } for point in points
    ]

def pluralize(word):
    if re.search('[sxz]$', word):
        return re.sub('$', 'es', word)
    elif re.search('[^aeioudgkprt]h$', word):
        return re.sub('$', 'es', word)
    elif re.search('[^aeiou]y$', word):
        return re.sub('y$', 'ies', word)
    else:
        return word + 's'

def format_question(points, count, label):
    return f"Can you please count the number of {pluralize(label.lower())} in the image?"

def format_answer(points, count, label):
    if count == 0:
        return f"There are no {pluralize(label.lower())} in the image."
    elif count == 1:
        return f"There is one {label} in the image."
    else:
        return f"There are {count} {pluralize(label.lower())} in the image."

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
            if cm == "counting":
                info["conversations"] = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "value": format_question(points, count, label)},
                            {"type": "image", "value": img_path}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "value": format_answer(points, count, label)}
                        ]
                    }
                ]
            information.append(json.dumps(info))

        return {
            "data": information,
            "files": [json.dumps([img_path]) for img_path in images]
        }