from vlm.config import supported_VLM
from vlm.dataset import build_dataset
from vlm.smp import *
import os
import argparse
import json
from tqdm import tqdm
import pandas as pd


def load_input_from_sample(sample):
    text = sample.get("question", "")
    choices = sample.get("answer_choices", [])
    if choices:
        choices_str = "\n".join([f"{c}" for c in choices])
        final_input = f"Text: {text}\nAnswer choices:\n{choices_str}"
    else:
        final_input = text
    return final_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Model name")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model")
    parser.add_argument('--search_model_path', type=str, required=True, help="Path to the search model")
    parser.add_argument('--zoom', action='store_true', help="Enable ZoomSearch mechanism")
    parser.add_argument('--save_intermediate', action='store_true', help="Save intermediate images")
    parser.add_argument('--output_dir', type=str, default='./outputs', help="Output directory")
    parser.add_argument('--dataset', type=str, default='MME-RealWorld',
                        help="Dataset name (default: MME-RealWorld)")
    parser.add_argument('--subset', type=str, default='RS',
                        help="Subset to evaluate (RS, Lite, etc.)")
    parser.add_argument('--reuse', action='store_true', help="Reuse existing predictions")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset_name = args.dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = build_dataset(dataset_name)

    if dataset is None:
        print(f"Failed to build dataset {dataset_name}")
        return

    data = dataset.data
    print(f"Total samples: {len(data)}")

    if args.subset:
        if 'category' in data.columns:
            rs_data = data[data['category'].str.contains(args.subset, case=False, na=False)]
        elif 'task' in data.columns:
            rs_data = data[data['task'].str.contains(args.subset, case=False, na=False)]
        elif 'image_path' in data.columns:
            rs_data = data[data['image_path'].str.contains(args.subset, case=False, na=False)]
        else:
            print(f"Warning: Cannot find subset '{args.subset}', using all data")
            rs_data = data

        if len(rs_data) == 0:
            print(f"No samples found for subset '{args.subset}', using all data")
            rs_data = data
        else:
            print(f"Filtered to {len(rs_data)} samples for subset '{args.subset}'")
        data = rs_data

    result_file = os.path.join(args.output_dir, f"{args.model}_{dataset_name}_{args.subset}.xlsx")

    existing_results = {}
    if args.reuse and os.path.exists(result_file):
        print(f"Loading existing results from {result_file}")
        existing_df = pd.read_excel(result_file)
        if 'index' in existing_df.columns and 'prediction' in existing_df.columns:
            existing_results = dict(zip(existing_df['index'], existing_df['prediction']))
            print(f"Loaded {len(existing_results)} existing predictions")

    print(f"Loading model: {args.model}")
    model = supported_VLM[args.model](
        model_path=args.model_path,
        max_new_tokens=1024,
        max_step=10,
        search_model_path=args.search_model_path,
        save_intermediate=args.save_intermediate
    )

    results = []
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Inferencing"):
        sample_idx = row.get('index', idx)

        if sample_idx in existing_results:
            prediction = existing_results[sample_idx]
        else:
            if 'image_path' in row:
                image_path = row['image_path']
            elif 'image' in row:
                image_path = row['image']
            else:
                print(f"Warning: No image path found for sample {sample_idx}")
                continue

            question = row.get('question', '')
            options = row.get('options', row.get('answer_choices', ''))

            if isinstance(options, str) and options:
                cur_input = f"{question}\n{options}"
            elif isinstance(options, list) and options:
                options_str = "\n".join(options)
                cur_input = f"{question}\nAnswer choices:\n{options_str}"
            else:
                cur_input = question

            try:
                prediction = model.generate(
                    [
                        dict(type='image', value=image_path),
                        dict(type='text', value=cur_input)
                    ],
                    zoom=args.zoom
                )
            except Exception as e:
                print(f"Error processing sample {sample_idx}: {e}")
                prediction = "ERROR"

        result = {
            'index': sample_idx,
            'question': row.get('question', ''),
            'answer': row.get('answer', row.get('ground_truth', '')),
            'prediction': prediction,
        }

        for col in ['category', 'task', 'image_path', 'image']:
            if col in row:
                result[col] = row[col]

        results.append(result)

        temp_df = pd.DataFrame(results)
        temp_df.to_excel(result_file, index=False)

    print(f"Final results saved to {result_file}")

    print("\n" + "="*50)
    print("Starting evaluation...")
    try:
        eval_results = dataset.evaluate(result_file)
        if eval_results is not None:
            print("Evaluation Results:")
            if isinstance(eval_results, dict):
                print(json.dumps(eval_results, indent=4))
            elif isinstance(eval_results, pd.DataFrame):
                print(eval_results.to_string())

            eval_file = os.path.join(args.output_dir, f"{args.model}_{dataset_name}_{args.subset}_eval.json")
            if isinstance(eval_results, dict):
                with open(eval_file, 'w') as f:
                    json.dump(eval_results, f, indent=4)
            else:
                eval_results.to_json(eval_file)
            print(f"Evaluation results saved to {eval_file}")
    except Exception as e:
        print(f"Evaluation failed: {e}")


if __name__ == '__main__':
    main()