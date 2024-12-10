from typing import Dict, List
import os
import json
from datasets import load_dataset
from utils.format import name_url
from utils.dataframe import BaseData, HFStaticDataIterator

def clean_question(text: str) -> str:
    """Clean text by removing leading and trailing whitespaces."""
    if text.startswith("[USER]"):
        text = text[len("[USER]"):]
    if text.endswith("[ASSISTANT]"):
        text = text[:-len("[ASSISTANT]")]
    return text.strip()

class Data(BaseData):
    info = "PIXMO Caption QA dataset containing image-based conversations"
    data_source = "pixmo_cap_qa"
    data_type = "conversation"
    length = 272000

    def load(self):
        self.image_folder_path = os.path.join(self.data_folder_path, "pixmo-images/images")
        return HFStaticDataIterator(load_dataset("allenai/pixmo-cap-qa", split="train"))

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
                            {"type": "text", "value": clean_question(q)}, 
                            {"type": "image", "value": img_path}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "value": a}
                        ]
                    }
                ]
            })
            for img_path, q, a in zip(images, batch["question"], batch["answer"])
        ]

        data = {
            "data": conversations,
            "files": [json.dumps([img_path]) for img_path in images]
        }
        return data