import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from ..base import BaseModel
from ...smp import *
import copy
import traceback
import warnings


class LLaVA(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    DEFAULT_IMAGE_TOKEN = '<image>'
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path='lmms-lab/llava-onevision-qwen2-7b-ov',
                 max_step=200,
                 bias_value=0.2, search_model_path='openbmb/VisRAG-Ret', save_intermediate=False, **kwargs):
        assert model_path is not None

        self.model_path = model_path
        self.model_type = self._detect_model_type(model_path)

        super().__init__(
            max_step=max_step,
            bias_value=bias_value,
            search_model_path=search_model_path,
            save_intermediate=save_intermediate
        )

        if self.model_type == 'llava_v1.5':
            self._init_llava_v15(model_path, **kwargs)
        elif self.model_type == 'llava_next_hf':
            self._init_llava_next_hf(model_path, **kwargs)
        elif self.model_type == 'llava_next2':
            self._init_llava_next2(model_path, **kwargs)
        elif self.model_type == 'llava_onevision':
            self._init_llava_onevision(model_path, **kwargs)
        elif self.model_type == 'llava_onevision_hf':
            self._init_llava_onevision_hf(model_path, **kwargs)
        else:
            self._init_llava_onevision(model_path, **kwargs)

        self.init_index_yes_no()

    def _detect_model_type(self, model_path):
        model_path_lower = model_path.lower()

        if 'llava-hf' in model_path_lower or 'llava-v1.6' in model_path_lower:
            if 'onevision' in model_path_lower:
                return 'llava_onevision_hf'
            return 'llava_next_hf'

        if 'onevision' in model_path_lower or 'llava-ov' in model_path_lower:
            return 'llava_onevision'

        if 'llava-next' in model_path_lower or 'llama3-llava' in model_path_lower:
            return 'llava_next2'

        if 'llava_v1.5' in model_path_lower or 'llava-v1.5' in model_path_lower or 'sharegpt4v' in model_path_lower:
            return 'llava_v1.5'

        return 'llava_onevision'

    def _init_llava_v15(self, model_path, **kwargs):
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, KeywordsStoppingCriteria
            from llava.conversation import conv_templates
        except ImportError:
            traceback.print_exc()
            warnings.warn('Please install llava from https://github.com/haotian-liu/LLaVA')
            raise

        self.system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )
        self.stop_str = "</s>"

        if model_path == 'Lin-Chen/ShareGPT4V-7B':
            model_name = 'llava-v1.5-7b'
        elif model_path == 'Lin-Chen/ShareGPT4V-13B':
            model_name = 'llava-v1.5-13b'
        else:
            model_name = get_model_name_from_path(model_path)

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path, model_base=None, model_name=model_name, device_map='cpu'
        )
        model = model.cuda().eval()

        self.conv_template = 'llava_v1'
        self.conv_templates = conv_templates
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.tokenizer_image_token = tokenizer_image_token
        self.process_images = process_images
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self.device = self.model.device

    def _init_llava_next_hf(self, model_path, **kwargs):
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoProcessor, LlavaForConditionalGeneration

        if '34b' in model_path.lower():
            self.processor = LlavaNextProcessor.from_pretrained(model_path, use_fast=False)
        elif 'interleave' in model_path.lower():
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            self.processor = LlavaNextProcessor.from_pretrained(model_path)

        flash_attn_flag = False
        try:
            import flash_attn
            flash_attn_flag = True
        except ImportError:
            pass

        model_cls = LlavaForConditionalGeneration if 'interleave' in model_path.lower() else LlavaNextForConditionalGeneration

        if flash_attn_flag:
            model = model_cls.from_pretrained(
                model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_flash_attention_2=True
            )
        else:
            model = model_cls.from_pretrained(
                model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
            )

        self.model = model.eval().cuda()
        self.tokenizer = self.processor.tokenizer
        self.device = self.model.device

    def _init_llava_next2(self, model_path, **kwargs):
        try:
            from llava.model.builder import load_pretrained_model
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.mm_utils import get_model_name_from_path, tokenizer_image_token, KeywordsStoppingCriteria
        except ImportError:
            traceback.print_exc()
            warnings.warn('Please `pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git`')
            raise

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name, device_map=None)
        model.cuda().eval()
        model.tie_weights()

        if 'llama3' in model_path.lower():
            conv_mode = 'llava_llama_3'
        elif 'qwen' in model_path.lower():
            conv_mode = 'qwen_1_5'
        else:
            conv_mode = 'llava_llama_3'

        self.conv_template = conv_mode
        self.conv_templates = conv_templates
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.tokenizer_image_token = tokenizer_image_token
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self.SeparatorStyle = SeparatorStyle
        self.device = self.model.device

    def _init_llava_onevision(self, model_path, **kwargs):
        try:
            from llava.model.builder import load_pretrained_model
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        except ImportError:
            traceback.print_exc()
            warnings.warn('Please `pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git`')
            raise

        model_name = "llava_qwen"
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name, device_map="auto"
        )
        model.eval()

        self.conv_template = 'qwen_1_5'
        self.conv_templates = conv_templates
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.tokenizer_image_token = tokenizer_image_token
        self.process_images = process_images
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
        self.SeparatorStyle = SeparatorStyle
        self.device = "cuda"

    def _init_llava_onevision_hf(self, model_path, **kwargs):
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to('cuda').eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.device = self.model.device

    def init_index_yes_no(self):
        yes_ids = self.tokenizer("Yes").input_ids
        no_ids = self.tokenizer("No").input_ids

        if len(yes_ids) == 1 and len(no_ids) == 1:
            self.index_yes = yes_ids[0]
            self.index_no = no_ids[0]
        elif len(yes_ids) >= 2 and len(no_ids) >= 2:
            self.index_yes = yes_ids[-1]
            self.index_no = no_ids[-1]
        else:
            self.index_yes = yes_ids[-1]
            self.index_no = no_ids[-1]

    @torch.no_grad()
    def get_confidence_value(self, content, image_list):
        if not isinstance(image_list, list):
            image_list = [image_list]

        if self.model_type == 'llava_next_hf':
            return self._get_confidence_llava_next_hf(content, image_list)
        elif self.model_type == 'llava_onevision_hf':
            return self._get_confidence_llava_onevision_hf(content, image_list)
        else:
            return self._get_confidence_llava_native(content, image_list)

    @torch.no_grad()
    def _get_confidence_llava_native(self, content, image_list):
        image_sizes = [img.size for img in image_list]
        image_tensor = self.process_images(image_list, self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device='cuda') for _image in image_tensor]

        full_content = self.DEFAULT_IMAGE_TOKEN + '\n' + content

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], full_content)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = self.tokenizer_image_token(prompt, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).to('cuda')

        outputs = self.model(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            return_dict=True
        )
        return self._cal_confidence(outputs)

    @torch.no_grad()
    def _get_confidence_llava_next_hf(self, content, image_list):
        content_list = [{"type": "image"}] * len(image_list) + [{"type": "text", "text": content}]
        conversation = [{"role": "user", "content": content_list}]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(prompt, image_list, return_tensors="pt").to("cuda", torch.float16)

        outputs = self.model(**inputs, return_dict=True)
        return self._cal_confidence(outputs)

    @torch.no_grad()
    def _get_confidence_llava_onevision_hf(self, content, image_list):
        content_with_images = self.DEFAULT_IMAGE_TOKEN * len(image_list) + "\n" + content
        conversation = [{"role": "user", "content": [{"type": "text", "text": content_with_images}]}]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image_list, text=prompt, return_tensors="pt").to('cuda', torch.float16)

        outputs = self.model(**inputs, return_dict=True)
        return self._cal_confidence(outputs)

    @torch.no_grad()
    def _cal_confidence(self, outputs):
        logits_yesno = outputs.logits[0, -1, [self.index_yes, self.index_no]]
        confidence = torch.softmax(logits_yesno, dim=-1)[0]
        confidence = 2 * (confidence.item() - 0.5)
        return confidence

    def generate_inner(self, message, dataset=None):
        if self.model_type == 'llava_v1.5':
            return self._generate_llava_v15(message, dataset)
        elif self.model_type == 'llava_next_hf':
            return self._generate_llava_next_hf(message, dataset)
        elif self.model_type == 'llava_next2':
            return self._generate_llava_next2(message, dataset)
        elif self.model_type == 'llava_onevision_hf':
            return self._generate_llava_onevision_hf(message, dataset)
        else:
            return self._generate_llava_onevision(message, dataset)

    def _generate_llava_v15(self, message, dataset=None):
        from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX

        content, images = '', []
        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
            else:
                content += ' <image> '
                img = msg['value'] if isinstance(msg['value'], Image.Image) else Image.open(msg['value']).convert('RGB')
                images.append(img)

        args = abstractproperty()
        args.image_aspect_ratio = 'pad'
        image_tensor = process_images(images, self.image_processor, args).to('cuda', dtype=torch.float16)

        prompt = self.system_prompt + "USER: " + content + " ASSISTANT: "
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=image_tensor, stopping_criteria=[stopping_criteria],
                do_sample=False, temperature=0, max_new_tokens=2048, top_p=None, num_beams=1, use_cache=True
            )

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output

    def _generate_llava_next_hf(self, message, dataset=None):
        content, images = [], []
        for msg in message:
            if msg['type'] == 'text':
                content.append({"type": "text", "text": msg['value']})
            else:
                content.append({"type": "image"})
                img = msg['value'] if isinstance(msg['value'], Image.Image) else Image.open(msg['value']).convert('RGB')
                images.append(img)

        conversation = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(prompt, images, return_tensors="pt").to("cuda", torch.float16)

        output = self.model.generate(**inputs, do_sample=False, temperature=0, max_new_tokens=2048, top_p=None, num_beams=1)
        answer = self.processor.decode(output[0], skip_special_tokens=True)
        answer = self._output_process_hf(answer)
        return answer

    def _generate_llava_next2(self, message, dataset=None):
        content, images = '', []
        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
            else:
                img = msg['value'] if isinstance(msg['value'], Image.Image) else Image.open(msg['value']).convert('RGB')
                images.append(img)
                content += self.DEFAULT_IMAGE_TOKEN + '\n'

        preprocess = self.image_processor.preprocess
        image_tensor = [preprocess(f, return_tensors='pt')['pixel_values'][0].half().cuda() for f in images]
        image_tensor = torch.stack(image_tensor)

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = self.tokenizer_image_token(prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        cont = self.model.generate(
            input_ids, images=image_tensor, do_sample=False, temperature=0,
            max_new_tokens=2048, stopping_criteria=[stopping_criteria]
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs

    def _generate_llava_onevision(self, message, dataset=None):
        content, images = '', []
        image_sizes = []

        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
            else:
                img = msg['value'] if isinstance(msg['value'], Image.Image) else Image.open(msg['value']).convert('RGB')
                images.append(img)
                image_sizes.append(img.size)

        question = self.DEFAULT_IMAGE_TOKEN + "\n" + content

        image_tensor = self.process_images(images, self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device='cuda') for _image in image_tensor]

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = self.tokenizer_image_token(prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).to('cuda')

        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs

    def _generate_llava_onevision_hf(self, message, dataset=None):
        content, images = '', []
        image_sizes = []

        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
            else:
                img = msg['value'] if isinstance(msg['value'], Image.Image) else Image.open(msg['value']).convert('RGB')
                images.append(img)
                image_sizes.append(img.size)
                content += self.DEFAULT_IMAGE_TOKEN + '\n'

        conversation = [{"role": "user", "content": [{"type": "text", "text": content}]}]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to('cuda', torch.float16)

        output = self.model.generate(**inputs, max_new_tokens=2048)
        return self.processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def _output_process_hf(self, answer):
        if '<s>' in answer:
            answer = answer.replace('<s>', '').strip()
        if '[/INST]' in answer:
            answer = answer.split('[/INST]')[1].strip()
        elif 'ASSISTANT:' in answer:
            answer = answer.split('ASSISTANT:')[1].strip()
        elif 'assistant\n' in answer:
            answer = answer.split('assistant\n')[1].strip()
        elif '<|end_header_id|>\n\n' in answer:
            answer = answer.split('<|end_header_id|>\n\n')[-1].strip()

        if '</s>' in answer:
            answer = answer.split('</s>')[0].strip()
        elif '<|im_end|>' in answer:
            answer = answer.split('<|im_end|>')[0].strip()
        elif '<|eot_id|>' in answer:
            answer = answer.split('<|eot_id|>')[0].strip()

        answer = answer.replace('<unk>', '')
        return answer