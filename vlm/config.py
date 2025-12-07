from vlm.vlm import *
from vlm.api import *
from functools import partial

llava_series = {
     # LLaVA v1.5 (llava-v1.5, sharegpt4v)
    'llava_v1.5_7b': partial(LLaVA, model_path='liuhaotian/llava-v1.5-7b'),
    'llava_v1.5_13b': partial(LLaVA, model_path='liuhaotian/llava-v1.5-13b'),
    'sharegpt4v_7b': partial(LLaVA, model_path='Lin-Chen/ShareGPT4V-7B'),
    'sharegpt4v_13b': partial(LLaVA, model_path='Lin-Chen/ShareGPT4V-13B'),

    # LLaVA OneVision (onevision, llava-ov)
    'llava_onevision_qwen2_0.5b_ov': partial(LLaVA, model_path='lmms-lab/llava-onevision-qwen2-0.5b-ov'),
    'llava_onevision_qwen2_7b_ov': partial(LLaVA, model_path='lmms-lab/llava-onevision-qwen2-7b-ov'),

    # LLaVA Next2 (llava-next, llama3-llava)
    'llama3_llava_next_8b': partial(LLaVA, model_path='lmms-lab/llama3-llava-next-8b'),

    # LLaVA Next HF (llava-hf, llava-v1.6)
    'llava_v1.6_mistral_7b_hf': partial(LLaVA, model_path='llava-hf/llava-v1.6-mistral-7b-hf'),
    'llava_v1.6_vicuna_7b_hf': partial(LLaVA, model_path='llava-hf/llava-v1.6-vicuna-7b-hf'),
    'llava_v1.6_vicuna_13b_hf': partial(LLaVA, model_path='llava-hf/llava-v1.6-vicuna-13b-hf'),

    # LLaVA OneVision HF (llava-hf + onevision)
    'llava_onevision_qwen2_7b_ov_hf': partial(LLaVA, model_path='llava-hf/llava-onevision-qwen2-7b-ov-hf')
}
qwen_series = {
    # Qwen2-VL
    'qwen2_vl_2b': partial(Qwen2VLChat, model_path='Qwen/Qwen2-VL-2B-Instruct'),
    'qwen2_vl_7b': partial(Qwen2VLChat, model_path='Qwen/Qwen2-VL-7B-Instruct'),
    # Qwen2.5-VL
    'qwen2.5_vl_3b': partial(Qwen2VLChat, model_path='Qwen/Qwen2.5-VL-3B-Instruct'),
    'qwen2.5_vl_7b': partial(Qwen2VLChat, model_path='Qwen/Qwen2.5-VL-7B-Instruct'),
    # Qwen2.5-Omni
    'qwen2.5_omni_7b': partial(Qwen2VLChat, model_path='Qwen/Qwen2.5-Omni-7B'),
}

supported_VLM = {}

model_groups = [
    llava_series,
    qwen_series
]

for grp in model_groups:
    supported_VLM.update(grp)
