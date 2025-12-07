from ..smp import *
from abc import abstractmethod
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from skimage.util import view_as_windows
import numpy as np
import math
import time
import traceback
import warnings
import os

from .layout_utils import (
    crop_image,
    place_crops_by_grid_layout,
)
from .search_models import create_search_model, BaseSearchModel

ANSWER_PROMPT = """Question: {}

Based on the current image crop shown:
- Is the target object/subject mentioned in the question present in this cropped region?
- Can you see the relevant visual content needed to answer this question?

Answer: Yes or No."""


def split_image(image, crop_size):
    width, height = image.size
    rest_w = crop_size - width % crop_size if width % crop_size > 0 else 0
    rest_h = crop_size - height % crop_size if height % crop_size > 0 else 0
    new_width = width + rest_w
    new_height = height + rest_h

    new_image = Image.new('RGB', (new_width, new_height), (128, 128, 128))
    new_image.paste(image, (0, 0))

    image_array = np.array(new_image)
    height, width, channels = image_array.shape

    patches = view_as_windows(
        image_array,
        window_shape=(crop_size, crop_size, channels),
        step=crop_size
    )
    rows, cols, _, _, _, _ = patches.shape
    image_patch_list = patches.reshape(-1, crop_size, crop_size, channels)

    image_list = [Image.fromarray(patch) for patch in image_patch_list]
    return image_list, rows, cols


class TreeNode:
    def __init__(self, answer_score=-1, retrieval_score=-1, depth=1, confidence=-1.,
                 bbox=None, parent=None, select_idx=None, eroded_image=None):
        self.depth = depth
        self.confidence = confidence
        self.bbox = bbox
        self.answer_score = answer_score
        self.retrieval_score = retrieval_score
        self.parent = parent
        self.select_idx = select_idx
        self.path_name = None
        self.position_info = None
        self.root_grid_position = None
        self.full_path = []
        self.eroded_image = eroded_image


def get_confidence_weight(cur_depth, bias_value=0.3):
    coeff = 1 - bias_value
    return coeff * (1 - (0.5 ** (cur_depth - 1))) + bias_value


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x * 5, -500, 500)))


def select_nodes_by_sigmoid(nodes, threshold=0.6, max_nodes=6):
    if len(nodes) == 0:
        return []

    if len(nodes) == 1:
        return nodes

    confidences = np.array([node.confidence for node in nodes])
    offset = np.mean(confidences)
    sigmoid_values = sigmoid(confidences - offset)
    selected_indices = np.where(sigmoid_values > threshold)[0]

    if len(selected_indices) == 0:
        best_idx = np.argmax(confidences)
        return [nodes[best_idx]]

    if len(selected_indices) > max_nodes:
        sorted_indices = selected_indices[np.argsort(sigmoid_values[selected_indices])[::-1]]
        selected_indices = sorted_indices[:max_nodes]

    return [nodes[i] for i in selected_indices]


def _infer_search_model_type(model_path: str) -> tuple:
    """
    Infer search model type from path.
    Returns: (model_type, model_name)
    """
    if model_path is None or model_path == '':
        return 'visrag', None

    path_lower = model_path.lower()

    if 'remoteclip' in path_lower:
        if 'rn50' in path_lower:
            return 'remoteclip', 'RN50'
        elif 'vit-b-32' in path_lower or 'vitb32' in path_lower:
            return 'remoteclip', 'ViT-B-32'
        elif 'vit-l-14' in path_lower or 'vitl14' in path_lower:
            return 'remoteclip', 'ViT-L-14'
        else:
            return 'remoteclip', 'ViT-L-14'

    if 'dgtrs' in path_lower:
        return 'dgtrs', None

    if 'rs5m' in path_lower:
        return 'rs5m', None

    return 'visrag', None


class BaseModel:
    INTERLEAVE = False
    allowed_types = ['text', 'image']

    def __init__(self, max_step=200, bias_value=0.2,
                 search_model_path='openbmb/VisRAG-Ret',
                 add_position_hint=True,
                 save_intermediate=False,
                 intermediate_image_path='process_image'):

        self.dump_image_func = None
        self.save_intermediate = save_intermediate
        self.intermediate_image_path = intermediate_image_path
        if self.save_intermediate:
            os.makedirs(self.intermediate_image_path, exist_ok=True)
            print(f"Intermediate images will be saved to: {self.intermediate_image_path}")

        self.search_model_path = search_model_path if search_model_path else 'openbmb/VisRAG-Ret'
        self._search_model: BaseSearchModel = None
        self._search_model_type, self._search_model_name = _infer_search_model_type(self.search_model_path)

        self.search_image_size = 224
        self.max_step = max_step
        self.bias_value = bias_value
        self.add_position_hint = add_position_hint
        self.keep_ratio = 0.5
        self._current_image_name = None
        self._intermediate_counter = 0


    def _load_search_model(self):
        """Lazily load the search model only when needed."""
        if self._search_model is not None and self._search_model.is_loaded:
            return True

        print(f"Initializing search model: {self._search_model_type} from {self.search_model_path}")

        if self._search_model_type == 'visrag':
            self._search_model = create_search_model('visrag', model_path=self.search_model_path)
        elif self._search_model_type == 'remoteclip':
            self._search_model = create_search_model('remoteclip', model_name=self._search_model_name, ckpt_path=self.search_model_path)
        elif self._search_model_type == 'dgtrs':
            self._search_model = create_search_model('dgtrs', ckpt_path=self.search_model_path)
        elif self._search_model_type == 'rs5m':
            self._search_model = create_search_model('rs5m', ckpt_path=self.search_model_path)
        else:
            raise ValueError(f"Unknown search model type: {self._search_model_type}")

        return self._search_model.load()

    @property
    def search_model(self) -> BaseSearchModel:
        """Property to access search model with lazy loading."""
        if self._search_model is None or not self._search_model.is_loaded:
            if not self._load_search_model():
                raise RuntimeError("Failed to load search model!")
        return self._search_model

    def _save_intermediate_image(self, image, stage, extra_info=""):
        if not self.save_intermediate:
            return

        self._intermediate_counter += 1
        filename = f"{self._current_image_name}_{self._intermediate_counter:04d}_{stage}"
        if extra_info:
            filename += f"_{extra_info}"
        filename += ".png"

        save_path = os.path.join(self.intermediate_image_path, filename)
        image.save(save_path)
        print(f"  [Saved] {save_path}")

    def _nine_grid_split_and_erosion(self, original_image, bbox, chunk_size, query_embedding, parent_depth=0):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        grid_width = width // 3
        grid_height = height // 3

        sub_regions = []
        position_info = []

        grid_positions = [
            'top-left', 'top-center', 'top-right',
            'middle-left', 'middle-center', 'middle-right',
            'bottom-left', 'bottom-center', 'bottom-right'
        ]

        for i in range(3):
            for j in range(3):
                grid_x1 = x1 + j * grid_width
                grid_y1 = y1 + i * grid_height

                if j == 2:
                    grid_x2 = x2
                else:
                    grid_x2 = grid_x1 + grid_width

                if i == 2:
                    grid_y2 = y2
                else:
                    grid_y2 = grid_y1 + grid_height

                grid_bbox = (grid_x1, grid_y1, grid_x2, grid_y2)
                grid_w = grid_x2 - grid_x1
                grid_h = grid_y2 - grid_y1

                if grid_w < chunk_size or grid_h < chunk_size:
                    continue

                grid_crop = crop_image(original_image, grid_bbox)

                try:
                    patches, rows, cols = split_image(grid_crop, chunk_size)
                except Exception as e:
                    continue

                if len(patches) == 0:
                    continue

                patch_embeddings = self.search_model.encode_images(patches)
                similarities = (query_embedding @ patch_embeddings.T).squeeze()
                similarities = torch.tensor(similarities) if not isinstance(similarities, torch.Tensor) else similarities
                similarities = 1 + similarities

                num_patches = len(patches)
                keep_k = max(1, int(self.keep_ratio * num_patches))

                if num_patches > 1:
                    topk_values, topk_indices = torch.topk(similarities, keep_k, largest=True, sorted=False)
                else:
                    topk_values = similarities
                    topk_indices = torch.tensor([0])

                retrieval_score = topk_values.mean().item()

                keep_mask = torch.zeros(num_patches, dtype=torch.bool)
                keep_mask[topk_indices] = True

                grid_array = np.array(grid_crop).copy()
                for patch_idx in range(num_patches):
                    if not keep_mask[patch_idx]:
                        row_idx = patch_idx // cols
                        col_idx = patch_idx % cols
                        patch_y1 = row_idx * chunk_size
                        patch_x1 = col_idx * chunk_size
                        patch_y2 = min(patch_y1 + chunk_size, grid_h)
                        patch_x2 = min(patch_x1 + chunk_size, grid_w)
                        grid_array[patch_y1:patch_y2, patch_x1:patch_x2] = 128

                if self.save_intermediate:
                    eroded_before_compress = Image.fromarray(grid_array.astype(np.uint8))
                    self._save_intermediate_image(
                        eroded_before_compress,
                        f"depth{parent_depth+1}_grid",
                        f"{grid_positions[i * 3 + j]}_before_compress"
                    )

                keep_mask_2d = keep_mask.reshape(rows, cols)

                if not keep_mask.any():
                    continue

                kept_rows = keep_mask_2d.any(dim=1)
                kept_cols = keep_mask_2d.any(dim=0)

                kept_row_indices = torch.where(kept_rows)[0].tolist()
                kept_col_indices = torch.where(kept_cols)[0].tolist()

                if len(kept_row_indices) == 0 or len(kept_col_indices) == 0:
                    continue

                compressed_h = len(kept_row_indices) * chunk_size
                compressed_w = len(kept_col_indices) * chunk_size
                compressed_array = np.zeros((compressed_h, compressed_w, 3), dtype=np.uint8)

                for new_row_idx, old_row_idx in enumerate(kept_row_indices):
                    for new_col_idx, old_col_idx in enumerate(kept_col_indices):
                        src_y1 = old_row_idx * chunk_size
                        src_y2 = min(src_y1 + chunk_size, grid_h)
                        src_x1 = old_col_idx * chunk_size
                        src_x2 = min(src_x1 + chunk_size, grid_w)

                        dst_y1 = new_row_idx * chunk_size
                        dst_x1 = new_col_idx * chunk_size

                        copy_h = src_y2 - src_y1
                        copy_w = src_x2 - src_x1

                        compressed_array[dst_y1:dst_y1+copy_h, dst_x1:dst_x1+copy_w] = \
                            grid_array[src_y1:src_y2, src_x1:src_x2]

                actual_h = (len(kept_row_indices) - 1) * chunk_size + min(chunk_size, grid_h - kept_row_indices[-1] * chunk_size)
                actual_w = (len(kept_col_indices) - 1) * chunk_size + min(chunk_size, grid_w - kept_col_indices[-1] * chunk_size)

                final_eroded_image = Image.fromarray(compressed_array[:actual_h, :actual_w].astype(np.uint8))

                if self.save_intermediate:
                    self._save_intermediate_image(
                        final_eroded_image,
                        f"depth{parent_depth+1}_grid",
                        f"{grid_positions[i * 3 + j]}_after_compress_score{retrieval_score:.3f}"
                    )

                final_bbox = grid_bbox

                if retrieval_score > 0:
                    sub_regions.append((final_bbox, retrieval_score, final_eroded_image))
                    position_info.append({
                        'position': grid_positions[i * 3 + j],
                        'bbox': final_bbox,
                        'grid_index': (i, j),
                        'kept_ratio': keep_k / num_patches,
                        'num_patches': num_patches,
                        'kept_patches': keep_k
                    })

        return sub_regions, position_info

    def _answer_confidence(self, query, image, bbox):
        crop = crop_image(image, bbox)
        prompt = ANSWER_PROMPT.format(query)
        answer_value = self.get_confidence_value(prompt, crop)
        return answer_value

    def _answer_confidence_with_image(self, query, image):
        prompt = ANSWER_PROMPT.format(query)
        answer_value = self.get_confidence_value(prompt, image)
        return answer_value

    def _build_position_hint(self):
        if not self.add_position_hint:
            return ""
        return "The input image has been padded with gray borders. When answering questions, the VLM should take these gray padding areas into account\n"

    def use_custom_prompt(self, dataset):
        return False

    @abstractmethod
    def build_prompt(self, line, dataset):
        raise NotImplementedError

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def preproc_content(self, inputs):
        from .file_utils import parse_file

        if self.check_content(inputs) == 'str':
            return [dict(type='text', value=inputs)]
        elif self.check_content(inputs) == 'dict':
            assert 'type' in inputs and 'value' in inputs
            return [inputs]
        elif self.check_content(inputs) == 'liststr':
            res = []
            for s in inputs:
                try:
                    mime, pth = parse_file(s)
                    if mime is None or mime == 'unknown':
                        res.append(dict(type='text', value=s))
                    else:
                        res.append(dict(type=mime.split('/')[0], value=pth))
                except:
                    res.append(dict(type='text', value=s))
            return res
        elif self.check_content(inputs) == 'listdict':
            for item in inputs:
                assert 'type' in item and 'value' in item
                try:
                    mime, s = parse_file(item['value'])
                    if mime is None:
                        assert item['type'] == 'text'
                    else:
                        assert mime.split('/')[0] == item['type']
                        item['value'] = s
                except:
                    pass
            return inputs
        else:
            return None

    @abstractmethod
    def generate_inner(self, message, dataset):
        raise NotImplementedError

    def check_content(self, msgs):
        if isinstance(msgs, str):
            return 'str'
        if isinstance(msgs, dict):
            return 'dict'
        if isinstance(msgs, list):
            types = [self.check_content(m) for m in msgs]
            if all(t == 'str' for t in types):
                return 'liststr'
            if all(t == 'dict' for t in types):
                return 'listdict'
        return 'unknown'

    def zoom_search(self, message):
        """
        Perform ZoomSearch using Greedy Best-First Search with sigmoid-based multi-node selection.
        """
        image = None
        query = None
        image_path = None

        for item in message:
            if item['type'] == 'image':
                image = item['value']
                if isinstance(image, str):
                    image_path = image
                    image = Image.open(image).convert("RGB")
            elif item['type'] == 'text':
                query = item['value']

        if image is None or query is None:
            return message

        if image_path:
            self._current_image_name = os.path.splitext(os.path.basename(image_path))[0]
        else:
            self._current_image_name = f"image_{int(time.time())}"
        self._intermediate_counter = 0

        if self.save_intermediate:
            current_save_dir = os.path.join(self.intermediate_image_path, self._current_image_name)
            os.makedirs(current_save_dir, exist_ok=True)
            original_intermediate_path = self.intermediate_image_path
            self.intermediate_image_path = current_save_dir
            self._save_intermediate_image(image, "original", "input")

        if self._search_model_type == 'visrag':
            INSTRUCTION = "Represent this query for retrieving relevant documents: "
            query_embedding = self.search_model.encode_text([INSTRUCTION + query])
        else:
            query_embedding = self.search_model.encode_text([query])

        img_w, img_h = image.size
        root_bbox = (0, 0, img_w, img_h)
        search_image_size = self.search_image_size

        root_node = TreeNode(
            depth=1,
            confidence=0,
            bbox=root_bbox,
            answer_score=0,
            retrieval_score=0,
            eroded_image=None
        )

        open_set = [root_node]
        leaf_nodes = []
        max_depth_reached = 1
        deepest_nodes = []

        num_pop = 0
        answering_confidence_threshold_upper = 1.0
        threshold_decrease = 0.1
        pop_num_limit = int(math.log(max(img_w, img_h) // search_image_size, 4) * 5)
        num_interval = 5
        visit_leaf = False

        max_step = self.max_step
        iteration = 0

        while len(open_set) > 0 and max_step > 0:
            iteration += 1
            f_value = [node.confidence for node in open_set]
            best_idx = np.argmax(f_value)
            cur_node = open_set.pop(best_idx)

            num_pop += 1
            max_step -= 1

            x1, y1, x2, y2 = cur_node.bbox
            cur_width = x2 - x1
            cur_height = y2 - y1

            is_leaf = max(cur_width, cur_height) <= search_image_size


            if is_leaf:
                visit_leaf = True
                leaf_nodes.append(cur_node)

                if self.save_intermediate and cur_node.eroded_image is not None:
                    path_str = "_".join(cur_node.full_path) if cur_node.full_path else "root"
                    self._save_intermediate_image(
                        cur_node.eroded_image,
                        f"leaf_depth{cur_node.depth}",
                        f"{path_str}_conf{cur_node.confidence:.3f}"
                    )

                if cur_node.depth > max_depth_reached:
                    max_depth_reached = cur_node.depth
                    deepest_nodes = [cur_node]
                elif cur_node.depth == max_depth_reached:
                    deepest_nodes.append(cur_node)

                continue

            if visit_leaf and cur_node.answer_score > answering_confidence_threshold_upper:
                break

            if num_pop >= pop_num_limit:
                answering_confidence_threshold_upper -= threshold_decrease
                pop_num_limit += num_interval

            try:
                sub_regions, position_info = self._nine_grid_split_and_erosion(
                    image, cur_node.bbox, search_image_size, query_embedding, parent_depth=cur_node.depth
                )
            except Exception as e:
                continue

            if len(sub_regions) == 0:
                continue

            expanded_nodes = []

            for idx, (sub_bbox, retrieval_score, eroded_image) in enumerate(sub_regions):
                answer_score = self._answer_confidence_with_image(query, eroded_image)

                w = get_confidence_weight(cur_node.depth, self.bias_value)
                confidence = (1. - w) * retrieval_score + w * answer_score

                new_node = TreeNode(
                    depth=cur_node.depth + 1,
                    confidence=confidence,
                    bbox=sub_bbox,
                    answer_score=answer_score,
                    retrieval_score=retrieval_score,
                    parent=cur_node,
                    select_idx=idx,
                    eroded_image=eroded_image
                )

                new_node.position_info = position_info[idx]

                if cur_node.depth == 1:
                    new_node.root_grid_position = position_info[idx]['position']
                    new_node.full_path = [position_info[idx]['position']]
                else:
                    new_node.root_grid_position = cur_node.root_grid_position
                    new_node.full_path = cur_node.full_path + [position_info[idx]['position']]

                expanded_nodes.append(new_node)

                if new_node.depth > max_depth_reached:
                    max_depth_reached = new_node.depth
                    deepest_nodes = [new_node]
                elif new_node.depth == max_depth_reached:
                    deepest_nodes.append(new_node)

            if len(expanded_nodes) > 0:
                selected_nodes = select_nodes_by_sigmoid(expanded_nodes, threshold=0.6, max_nodes=6)
                open_set.extend(selected_nodes)

        final_candidates = list(set(leaf_nodes + deepest_nodes))

        if len(final_candidates) == 0:
            final_selected = [root_node]
        else:
            final_selected = select_nodes_by_sigmoid(final_candidates, threshold=0.6, max_nodes=6)

        if self.save_intermediate:
            print(f"\n[Final Selection] {len(final_selected)} nodes selected")
            for i, node in enumerate(final_selected):
                if node.eroded_image is not None:
                    path_str = "_".join(node.full_path) if node.full_path else "root"
                    self._save_intermediate_image(
                        node.eroded_image,
                        f"final_selected_{i}",
                        f"{path_str}_conf{node.confidence:.3f}"
                    )

        new_image = place_crops_by_grid_layout(image, final_selected)

        if self.save_intermediate:
            self._save_intermediate_image(new_image, "final_layout", "output")
            self.intermediate_image_path = original_intermediate_path
            print(f"[ZoomSearch] Done. Total intermediate images saved: {self._intermediate_counter}")

        position_hint = self._build_position_hint()

        for mess in message:
            if mess['type'] == 'image':
                mess['value'] = new_image
            elif mess['type'] == 'text':
                mess['value'] = f"{position_hint}Question:\n{mess['value']}"

        return message

    def generate(self, message, dataset=None, zoom=False):
        """
        Generate the output message.
        If zoom=True, performs ZoomSearch before generation.
        """
        assert self.check_content(message) in ['str', 'dict', 'liststr', 'listdict'], f'Invalid input type: {message}'
        for item in message:
            assert item['type'] in self.allowed_types, f'Invalid input type: {item["type"]}'

        if zoom:
            message = self.zoom_search(message)

        return self.generate_inner(message, dataset)