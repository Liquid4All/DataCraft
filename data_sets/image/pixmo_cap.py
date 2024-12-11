from utils.dataframe import BaseData, HFStaticDataIterator
import os
import json
from utils.format import name_url
from datasets import load_dataset

class Data(BaseData):
    data_source = "pixmo_cap"
    def load(self):
        self.image_folder_path = os.path.join(self.data_folder_path, "pixmo-cap-images")
        if not os.path.exists(self.image_folder_path):
            os.makedirs(self.image_folder_path)
        return HFStaticDataIterator(load_dataset('allenai/pixmo-cap', split='train'))

    def batch_process(self, examples):
        urls = examples["image_url"]
        paths = [os.path.join(self.image_folder_path, name_url(url)) for url in urls]
        examples["path"] = paths
        
        for url, path in zip(urls, paths):
            self.add_file(url, path)
            
        data = {"data": [json.dumps({"text": cap}) for cap in examples["caption"]], "files": [json.dumps([path]) for path in paths]}
        return data