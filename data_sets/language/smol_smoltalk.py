from utils.dataframe import BaseData, HFStaticDataIterator
import json
from datasets import load_dataset
from data_sets.language.smoltalk import convert_to_multimodal_format

class Data(BaseData):
    info = "SmolTalk dataset containing conversation data"
    data_type = "conversation"
    data_source = "smol_smoltalk"
    
    def load(self):
        return HFStaticDataIterator(load_dataset('HuggingFaceTB/smol-smoltalk', split='train'))

    def batch_process(self, examples):
        return {
            "data": [json.dumps(convert_to_multimodal_format(examples["messages"][i])) for i in range(len(examples["messages"]))]
        }