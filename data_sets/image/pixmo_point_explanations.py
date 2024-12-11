from typing import Dict, List
import os
import json
from datasets import load_dataset
from utils.format import name_url
from utils.dataframe import BaseData, HFStaticDataIterator

def format_points(labels, texts, points):
    if len(points) == 0:
        return {
            "points": []
        }
    return [
        {
            "label": f"{label} ({txt})",
            "coordinates": [x, y]
        }
        for [x, y], label, txt in zip(points[0], labels, texts)
    ]

class Data(BaseData):
    info = "PIXMO Point Explanations dataset containing image-based conversations"
    data_source = "pixmo_point_explanations"
    data_type = "conversation"
    length = 272000

    def load(self):
        self.image_folder_path = os.path.join(self.data_folder_path, "pixmo-images/images")
        return HFStaticDataIterator(load_dataset("allenai/pixmo-point-explanations", split="train"))

    def batch_process(self, batch: Dict[str, List]) -> Dict[str, List]:
        images = [os.path.join(self.image_folder_path, name_url(url)) for url in batch["image_url"]]
        for url, path in zip(batch["image_url"], images):
            self.add_file(url, path)
        
        conversations = [
            json.dumps({
                "conversations": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "value": q}, 
                            {"type": "image", "value": img_path}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "value": a}
                        ]
                    }
                ],
                "points": format_points(labels, texts, points)
            })
            for img_path, q, a, labels, texts, points in zip(images, batch["question"], batch["response"], batch["alt_text"], batch["inline_text"], batch["points"])
        ]
        
        return {
            "data": conversations,
            "files": [json.dumps([img_path]) for img_path in images]
        }