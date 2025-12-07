import warnings
import re
import json
import os
from tqdm import tqdm

from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *


def LOCALIZE(fname, new_fname=None):
    if new_fname is None:
        new_fname = fname.replace('.tsv', '_local.tsv')

    base_name = osp.basename(fname)
    dname = osp.splitext(base_name)[0]

    data = load(fname)
    data_new = localize_df(data, dname)
    dump(data_new, new_fname)
    print(f'The localized version of data file is {new_fname}')
    return new_fname


class ImageMCQDataset(ImageBaseDataset):

    TYPE = 'MCQ'

    DATASET_URL = {}
    DATASET_MD5 = {}

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above.\n'

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import report_acc, mcq_vanilla_eval

        dataset = self.dataset_name
        nproc = judge_kwargs.pop('nproc', 4)

        suffix = eval_file.split('.')[-1]
        model = judge_kwargs.get('model', 'exact_matching')
        assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        elif gpt_key_set():
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None
        else:
            warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
            model = None

        result_file = eval_file.replace(f'.{suffix}', f'_{name_str}_result.pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)

        dump(data, eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))
        data = load(eval_file.replace(f'.{suffix}', f'_{name_str}_result.{suffix}'))

        acc = report_acc(data)
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(acc, score_file)

        return acc


class MMERealWorld(ImageMCQDataset):

    TYPE = 'MMERealWorld'

    DATASET_MD5 = {
        'MME-RealWorld': 'fd341132362fc41e707cb37ee6e4dcda',
        'MME-RealWorld-Lite': '4c17057d7d3b6c4a0d4397c3dae0881c',
        'MME-RealWorld-CN': 'daaa763d52a760a38606d5dedb3fe444',
    }

    SYS = {
        'MME-RealWorld': (
            'Select the best answer to the above multiple-choice question based on the image. '
            'Respond with only the letter (A, B, C, D, or E) of the correct option. \n'
            'The best answer is:'
        ),
        'MME-RealWorld-Lite': (
            'Select the best answer to the above multiple-choice question based on the image. '
            'Respond with only the letter (A, B, C, D, or E) of the correct option. \n'
            'The best answer is:'
        ),
        'MME-RealWorld-CN': (
            '根据图像选择上述多项选择题的最佳答案。只需回答正确选项的字母（A, B, C, D 或 E）。\n'
            '最佳答案为：'
        ),
    }

    @classmethod
    def supported_datasets(cls):
        return ['MME-RealWorld', 'MME-RealWorld-CN', 'MME-RealWorld-Lite']

    def load_data(self, dataset="MME-RealWorld", repo_id="yifanzhang114/MME-RealWorld-Base64"):

        def check_integrity(pth):
            data_file = osp.join(pth, f"{dataset}.tsv")
            if not os.path.exists(data_file):
                return False
            if md5(data_file) != self.DATASET_MD5[dataset]:
                return False
            return True

        def generate_tsv(pth):
            tsv_file = os.path.join(pth, f"{dataset}.tsv")
            if os.path.exists(tsv_file):
                print(f"{tsv_file} already exists.")
                return

            json_dir = os.path.join(pth, dataset)
            json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

            data_list = []
            for json_file in json_files:
                with open(os.path.join(json_dir, json_file), "r") as f:
                    data = json.load(f)
                    for item in tqdm(data):
                        choice_prompt = (
                            "The choices are listed below:\n"
                            if dataset in ["MME-RealWorld", "MME-RealWorld-Lite"]
                            else "选项如下所示:\n"
                        )
                        data_list.append({
                            "index": item["index"],
                            "image": item["image"],
                            "question": item["question"],
                            "multi-choice options": choice_prompt + "\n".join(item["multi-choice options"]),
                            "A": item["multi-choice options"][0][4:],
                            "B": item["multi-choice options"][1][4:],
                            "C": item["multi-choice options"][2][4:],
                            "D": item["multi-choice options"][3][4:],
                            "E": item["multi-choice options"][4][4:],
                            "answer": item["answer"],
                            "category": item["category"],
                            "l2-category": item["l2-category"],
                        })
            df = pd.DataFrame(data_list)
            df.to_csv(tsv_file, sep="\t", index=False)
            print(f"TSV file saved to {tsv_file}")

        if dataset == "MME-RealWorld-Lite":
            url = 'https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Base64/resolve/main/mme_realworld_lite.tsv'
            file_md5 = self.DATASET_MD5[dataset] if dataset in self.DATASET_MD5 else None
            datas = self.prepare_tsv(url, file_md5)
            choice_prompt = "The choices are listed below:\n"
            for index, item in datas.iterrows():
                options = eval(item["multi-choice options"])
                datas.loc[index, "multi-choice options"] = choice_prompt + "\n".join(options)
                datas.loc[index, "A"] = options[0][4:]
                datas.loc[index, "B"] = options[1][4:]
                datas.loc[index, "C"] = options[2][4:]
                datas.loc[index, "D"] = options[3][4:]
                datas.loc[index, "E"] = options[4][4:]
            return datas

        update_flag = False
        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
            print(f"Using cached dataset from {cache_path}")
        else:
            from huggingface_hub import snapshot_download
            dataset_path = snapshot_download(repo_id=repo_id, repo_type="dataset")
            generate_tsv(dataset_path)
            update_flag = True

        data_path = os.path.join(dataset_path, f"{dataset}.tsv")
        if file_size(data_path, "GB") > 1:
            local_path = data_path.replace(".tsv", "_local.tsv")
            if not osp.exists(local_path) or os.environ.get("FORCE_LOCAL", None) or update_flag:
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)

    def post_build(self, dataset):
        self.TYPE = 'MMERealWorld'

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        choice_prompt = line['multi-choice options'] + '\n'
        question += ' ' + choice_prompt + self.SYS[self.dataset_name]

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        from .utils.multiple_choice import extract_characters_regex, get_dimension_rating

        assert eval_file.split('.')[-1] in ['xlsx', 'json', 'tsv']
        FAIL_MSG = 'Failed to obtain answer via API.'
        suffix = eval_file.split('.')[-1]
        tmp_file = eval_file.replace(f'.{suffix}', '_tmp.pkl')
        tgt_file = eval_file.replace(f'.{suffix}', '_rating.json')
        score_file = eval_file.replace(f'.{suffix}', '_score.xlsx')

        if not osp.exists(score_file):
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            cnt_rejected = 0
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = data.loc[data['index'] == idx, 'prediction'].values[0]

                match_pred = re.search(r"<answer>(.*?)</answer>", pred, re.DOTALL)
                pred = match_pred.group(1).strip().upper() if match_pred else pred

                extract_pred = extract_characters_regex(pred)
                if extract_pred == '':
                    cnt_rejected += 1
                    data.loc[data['index'] == idx, 'score'] = 0
                else:
                    data.loc[data['index'] == idx, 'score'] = int(extract_pred == ans)

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {cnt_rejected} questions. '
                f'Those questions will be counted as 0 score in ALL rating.'
            )
            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating


class CustomMCQDataset(ImageMCQDataset):

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)