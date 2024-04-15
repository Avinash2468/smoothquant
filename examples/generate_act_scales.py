import torch
import os
import sys

sys.path.insert(1, '/storage/ice1/7/4/apuppala6/project/smoothquant')

# from transformers import (
#     AutoModelForCausalLM,
#     AutoTokenizer,
# )

from transformers import (
    AutoProcessor, 
    LlavaForConditionalGeneration,
)

from PIL import Image
import requests

import argparse

from smoothquant.calibration import get_act_scales

# Modified this
def build_model_and_tokenizer(model_name):
    tokenizer = AutoProcessor.from_pretrained(model_name, model_max_length=512, cache_dir = "/storage/ice1/7/4/apuppala6/hugging_face_cache")
    print("\n LOADED tokenizer \n")
    kwargs = {"torch_dtype": torch.float16, "device_map": "auto", "cache_dir": "/storage/ice1/7/4/apuppala6/hugging_face_cache"}
    model = LlavaForConditionalGeneration.from_pretrained(model_name, **kwargs)
    print("\n LOADED model \n")
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", type=str, default="llava-hf/llava-1.5-7b-hf", help="model name"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="../act_scales/llava-1.5-7b-hf",
        help="where to save the act scales",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="dataset/val.jsonl.zst",
        help="location of the calibration dataset, we use the validation set of the Pile dataset",
    )
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, processor = build_model_and_tokenizer(args.model_name)

    # if not os.path.exists(args.dataset_path):
    #     print(f"Cannot find the dataset at {args.dataset_path}")
    #     print("Please download the Pile dataset and put the validation set at the path")
    #     print(
    #         "You can download the validation dataset of the Pile at https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst"
    #     )
    #     raise FileNotFoundError

    prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
    url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    print("downloaded Image")
    act_scales = get_act_scales(
        model, processor, prompt, image
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)


if __name__ == "__main__":
    main()
