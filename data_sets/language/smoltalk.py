from utils.dataframe import BaseData, HFStaticDataIterator
import json
from datasets import load_dataset

SMOLTALK_SUBSETS = ["all", "apigen-80k", "everyday-conversations", "explore-instruct-rewriting", "longalign", "metamathqa-50k",  "numina-cot-100k", "openhermes-100k", "self-oss-instruct", "smol-constraints", "smol-magpie-ultra", "smol-rewrite", "smol-summarize", "systemchats-30k"]

def convert_to_multimodal_format(messages):
    """Convert messages to multi-modal conversation format."""
    return {
        "conversations": [
            {
                "role": msg["role"],
                "content": [
                    {
                        "type": "text",
                        "value": msg["content"]
                    }
                ]
            }
            for msg in messages
        ]
    }

class Data(BaseData):
    info = "SmolTalk dataset containing conversation data"
    data_type = "conversation"

    def __init__(self):
        assert self.dataset_subset is not None, f"""Must provide dataset_subset, choose from {self.dataset_subset}"""
        assert self.dataset_subset in SMOLTALK_SUBSETS, f"""got {self.dataset_subset} for dataset_subset, but must be one of {str(SMOLTALK_SUBSETS)}"""
        self.data_source = "smoltalk_" + self.dataset_subset

    def load(self):
        return HFStaticDataIterator(load_dataset('HuggingFaceTB/smoltalk', self.dataset_subset, split='train'))

    def batch_process(self, examples):
        return {
            "data": [json.dumps(convert_to_multimodal_format(examples["messages"][i])) for i in range(len(examples["messages"]))]
        }