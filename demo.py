from vlm.config import supported_VLM
import os
import traceback
import time
from PIL import Image
import argparse
import json

def load_input_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    text = data.get("text", "")
    choices = data.get("answer_choices", [])
    choices_str = "\n".join([f"{c}" for c in choices])

    final_input = f"Text: {text}\nAnswer choices:\n{choices_str}"
    return final_input


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Model name")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model")
    parser.add_argument('--search_model_path', type=str, required=True, help="Path to the search model")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image")
    parser.add_argument('--json_path', type=str, help="Path to the JSON file with questions/choices")
    parser.add_argument('--zoom', action='store_true', help="Enable ZoomSearch mechanism")
    parser.add_argument('--input', type=str, help="Text input/question for the model")
    parser.add_argument('--save_intermediate', action='store_true', help="Save intermediate images")

    args = parser.parse_args()
    if args.json_path and os.path.exists(args.json_path):
        cur_input = load_input_from_json(args.json_path)
    else:
        cur_input = args.input

    model = supported_VLM[args.model](
        model_path=args.model_path,
        max_new_tokens=1024,
        max_step=10,
        search_model_path=args.search_model_path,
        save_intermediate=args.save_intermediate
    )

    ret = model.generate([
            dict(type='image', value=args.image_path),
            dict(type='text', value=cur_input)
        ],
        zoom=args.zoom
    )
    print(f"Image_path: {args.image_path}\nInput:\n{cur_input}\n\nResponse: {ret}")
