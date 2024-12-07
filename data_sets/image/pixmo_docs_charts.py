from utils.dataframe import BaseData, HFStaticDataIterator
import os
import json
from datasets import load_dataset

class Data(BaseData):
    data_source = "pixmo_docs_charts"
    pixmo_docs_group = "charts"
    def load(self):
        self.image_folder_path = os.path.join(self.data_folder_path, "pixmo-docs-images")
        if not os.path.exists(self.image_folder_path):
            os.makedirs(self.image_folder_path)
        return HFStaticDataIterator(load_dataset('allenai/pixmo-docs', self.pixmo_docs_group, split='train'))

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
                                {
                                    "type": "text",
                                    "value": str(q)
                                }
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "value": str(a)
                                }
                            ]
                        }
                    ]
                    for q, a in zip(item["question"], item["answer"])
                })
                for item in examples["questions"]
            ],
            "files": [json.dumps([image_path]) for image_path in paths]
        }
        return data