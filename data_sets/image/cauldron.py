from utils.dataframe import BaseData, HFStaticDataIterator
import os
import json
from datasets import load_dataset

CAULDRON_SUBSETS = ["ai2d", "aokvqa", "chart2text", "chartqa", "clevr", "clevr_math", "cocoqa", "datikz", "diagram_image_to_text", "docvqa", "dvqa", "figureqa", "finqa", "geomverse", "hateful_memes", "hitab", "iam", "iconqa", "infographic_vqa", "intergps", "localized_narratives", "mapqa", "mimic_cgd", "multihiertt", "nlvr2", "ocrvqa", "okvqa", "plotqa", "raven", "rendered_text", "robut_sqa", "robut_wikisql", "robut_wtq", "scienceqa", "screen2words", "spot_the_diff", "st_vqa", "tabmwp", "tallyqa", "tat_qa", "textcaps", "textvqa", "tqa", "vistext", "visual7w", "visualmrc", "vqarad", "vqav2", "vsr", "websight"]

def format_conversations(convs, images):
    image_content = [{"type": "image", "value": img} for img in images]
    formatted_convs = [
        {
            "role": "user", 
            "content": image_content + [
                {"type": "text", "value": convs[0]["user"]}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "value": convs[0]["assistant"]}
            ]
        }
    ]
    for conv in convs[1:]:
        formatted_convs.append(
            {
                "role": "user", 
                "content": [
                    {"type": "text", "value": conv["user"]}
                ]
            }
        )
        formatted_convs.append(
            {
                "role": "assistant", 
                "content": [
                    {"type": "text", "value": conv["assistant"]}
                ]
            }
        )
    return formatted_convs

class Data(BaseData):
    info = "The Cauldron dataset containing image-based conversations"
    data_type = "conversation"

    def __init__(self):
        assert self.dataset_subset is not None, f"""Must provide dataset_subset, choose from {self.dataset_subset}"""
        assert self.dataset_subset in CAULDRON_SUBSETS, f"""got {self.dataset_subset} for dataset_subset, but must be one of {str(CAULDRON_SUBSETS)}"""
        self.data_source = "cauldron_" + self.dataset_subset

    def load(self):
        self.image_folder_path = os.path.join(self.data_folder_path, "llava-cauldron-images") # Saves to images folder
        if not os.path.exists(self.image_folder_path):
            os.makedirs(self.image_folder_path)
        dataset = load_dataset('HuggingFaceM4/the_cauldron', self.dataset_subset, split='train')

        def add_ids(example, idx):
            example["id"] = f"{self.dataset_subset}_{idx}"
            return example

        return HFStaticDataIterator(dataset.map(add_ids, with_indices=True))

    def batch_process(self, examples):        
        images = examples["images"]
        
        paths = []
        for imgs, row_id in zip(images, examples["id"]):
            img_group = []
            for i, img in enumerate(imgs):
                path = os.path.join(self.image_folder_path, f"{self.dataset_subset}_{row_id}_{i}.png")
                self.add_file(img, path)
                img_group.append(path)
            paths.append(img_group)
            
        return {
            "data": [
                json.dumps({"conversations": format_conversations(convs, imgs)}) for convs, imgs in zip(examples["texts"], paths)
            ],
            "files": [json.dumps(path_groups) for path_groups in paths]
        }