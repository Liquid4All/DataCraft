from utils.dataframe import BaseData, HFStaticDataIterator
import os
import json
from datasets import load_dataset
from utils.format import image_conversation

ONEVISION_SUBSETS = ["CLEVR-Math(MathV360K)", "FigureQA(MathV360K)", "GEOS(MathV360K)", "GeoQA+(MathV360K)", "Geometry3K(MathV360K)", "IconQA(MathV360K)", "MapQA(MathV360K)", "PMC-VQA(MathV360K)", "Super-CLEVR(MathV360K)", "TabMWP(MathV360K)", "UniGeo(MathV360K)", "VisualWebInstruct(filtered)", "VizWiz(MathV360K)", "ai2d(cauldron,llava_format)", "ai2d(gpt4v)", "ai2d(internvl)", "allava_instruct_laion4v", "allava_instruct_vflan4v", "aokvqa(cauldron,llava_format)", "chart2text(cauldron)", "chartqa(cauldron,llava_format)", "chrome_writting", "clevr(cauldron,llava_format)", "diagram_image_to_text(cauldron)", "dvqa(cauldron,llava_format)", "figureqa(cauldron,llava_format)", "geo170k(align)", "geo170k(qa)", "geo3k", "geomverse(cauldron)", "hateful_memes(cauldron,llava_format)", "hitab(cauldron,llava_format)", "hme100k", "iam(cauldron)", "iconqa(cauldron,llava_format)", "iiit5k", "image_textualization(filtered)", "infographic(gpt4v)", "infographic_vqa", "infographic_vqa_llava_format", "intergps(cauldron,llava_format)", "k12_printing", "llavar_gpt4_20k", "lrv_chart", "lrv_normal(filtered)", "magpie_pro(l3_80b_mt)", "magpie_pro(l3_80b_st)", "magpie_pro(qwen2_72b_st)", "mapqa(cauldron,llava_format)", "mathqa", "mavis_math_metagen", "mavis_math_rule_geo", "multihiertt(cauldron)", "orand_car_a", "raven(cauldron)", "rendered_text(cauldron)", "robut_sqa(cauldron)", "robut_wikisql(cauldron)", "robut_wtq(cauldron,llava_format)", "scienceqa(cauldron,llava_format)", "scienceqa(nona_context)", "screen2words(cauldron)", "sharegpt4o", "sharegpt4v(coco)", "sharegpt4v(knowledge)", "sharegpt4v(llava)", "sharegpt4v(sam)", "sroie", "st_vqa(cauldron,llava_format)", "tabmwp(cauldron)", "tallyqa(cauldron,llava_format)", "textcaps", "textocr(gpt4v)", "tqa(cauldron,llava_format)", "ureader_cap", "ureader_ie", "vision_flan(filtered)", "vistext(cauldron)", "visual7w(cauldron,llava_format)", "visualmrc(cauldron)", "vqarad(cauldron,llava_format)", "vsr(cauldron,llava_format)", "websight(cauldron)"]

def clean_question(question):
    if question.startswith("<image>"):
        return question.replace("<image>", "").lstrip()
    return question

class Data(BaseData):
    info = "LLaVA-OneVision dataset containing image-based conversations"
    data_type = "conversation"

    def __init__(self):
        assert self.dataset_subset is not None, f"""Must provide dataset_subset, choose from {self.dataset_subset}"""
        assert self.dataset_subset in ONEVISION_SUBSETS, f"""got {self.dataset_subset} for dataset_subset, but must be one of {str(ONEVISION_SUBSETS)}"""
        self.data_source = "llava_onevision_" + self.dataset_subset

    def load(self):
        self.image_folder_path = os.path.join(self.data_folder_path, "llava-onevision-images") # Saves to images folder
        if not os.path.exists(self.image_folder_path):
            os.makedirs(self.image_folder_path)
        return HFStaticDataIterator(load_dataset('lmms-lab/LLaVA-OneVision-Data', self.dataset_subset, split='train'))

    def batch_process(self, examples):
        # Filter to only include conversations with at least 2 messages
        valid_indices = [i for i, convs in enumerate(examples["conversations"]) if len(convs) >= 2]
        examples = {
            key: [examples[key][i] for i in valid_indices] 
            for key in examples
        }
        
        images = examples["image"]
        paths = [os.path.join(self.image_folder_path, self.dataset_subset + "_" + str(image_id) + ".png") for image_id in examples["id"]]
        for img, path in zip(images, paths):
            self.add_file(img, path)
        return {
            "data": [
                json.dumps({
                    "conversations": image_conversation(clean_question(conv[0]["value"]), conv[1]["value"], [img])
                }) for conv, img in zip(examples["conversations"], paths)
            ],
            "files": [json.dumps([image_path]) for image_path in paths]
        }