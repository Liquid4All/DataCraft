from utils.dataframe import BaseData, HFStaticDataIterator
import os
import json
from datasets import load_dataset

class Data(BaseData):
    def __init__(self):
        assert self.dataset_subset is not None, """Must provide dataset_subset, choose from ["charts", "diagrams", "other", "tables"]"""
        assert self.dataset_subset in ["charts", "diagrams", "other", "tables"], f"""dataset_subset must be one of ["charts", "diagrams", "other", "tables"], got {self.dataset_subset}"""
        self.data_source = "pixmo_docs_" + self.dataset_subset

    def load(self):
        self.image_folder_path = os.path.join(self.data_folder_path, "pixmo-images") # Saves to images folder
        if not os.path.exists(self.image_folder_path):
            os.makedirs(self.image_folder_path)
        return HFStaticDataIterator(load_dataset('allenai/pixmo-docs', self.dataset_subset, split='train'))

    def batch_process(self, examples):
        images = examples["image"]
        paths = [os.path.join(self.image_folder_path, str(image_id)) for image_id in examples["image_id"]]
        for img, path in zip(images, paths):
            self.add_file(img, path)
        data = {
            "data": [
                json.dumps({
                    "conversations": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "value": img},
                                {"type": "text", "value": q}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "value": a}
                            ]
                        }
                    ]
                    for q, a, img in zip(item["question"], item["answer"], paths)
                })
                for item in examples["questions"]
            ],
            "files": [json.dumps([image_path]) for image_path in paths]
        }
        return data