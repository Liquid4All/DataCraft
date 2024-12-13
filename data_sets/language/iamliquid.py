from utils.dataframe import BaseData, HFStaticDataIterator
import json
from datasets import load_dataset
from utils.format import chatml_to_conversation

class Data(BaseData):
    info = "Liquid AI dataset for identity training"
    data_type = "conversation"
    data_source = "iamliquid"
    
    def load(self):
        return HFStaticDataIterator(load_dataset('LiquidAI/iamliquid-instruction', split='train'))

    def batch_process(self, examples):
        return {
            "data": [json.dumps(chatml_to_conversation(row)) for row in examples["conversations"]]
        }