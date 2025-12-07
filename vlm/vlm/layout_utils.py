from PIL import Image
import numpy as np

POS_TO_GRID = {
    'top-left': (0, 0), 'top-center': (0, 1), 'top-right': (0, 2),
    'middle-left': (1, 0), 'middle-center': (1, 1), 'middle-right': (1, 2),
    'bottom-left': (2, 0), 'bottom-center': (2, 1), 'bottom-right': (2, 2)
}


def crop_image(image, bbox):
    x1, y1, x2, y2 = bbox
    return image.crop((x1, y1, x2, y2))


def _get_root_grid_bbox(original_image, root_grid_position):
    img_w, img_h = original_image.size
    grid_w = img_w // 3
    grid_h = img_h // 3

    if root_grid_position not in POS_TO_GRID:
        return (0, 0, img_w, img_h)

    row, col = POS_TO_GRID[root_grid_position]

    x1 = col * grid_w
    y1 = row * grid_h
    x2 = img_w if col == 2 else (col + 1) * grid_w
    y2 = img_h if row == 2 else (row + 1) * grid_h

    return (x1, y1, x2, y2)


def _simple_concat_patches(patches):
    if len(patches) == 0:
        return Image.new('RGB', (100, 100), (128, 128, 128))
    if len(patches) == 1:
        return patches[0]

    max_h = max(p.size[1] for p in patches)
    resized_patches = []
    for p in patches:
        if p.size[1] != max_h:
            new_w = int(p.size[0] * max_h / p.size[1])
            resized_patches.append(p.resize((new_w, max_h), Image.LANCZOS))
        else:
            resized_patches.append(p)

    total_w = sum(p.size[0] for p in resized_patches)
    canvas = Image.new('RGB', (total_w, max_h), (128, 128, 128))

    x_offset = 0
    for p in resized_patches:
        canvas.paste(p, (x_offset, 0))
        x_offset += p.size[0]

    return canvas


def _vertical_concat_patches(patches):
    if len(patches) == 0:
        return Image.new('RGB', (100, 100), (128, 128, 128))
    if len(patches) == 1:
        return patches[0]

    max_w = max(p.size[0] for p in patches)
    resized_patches = []
    for p in patches:
        if p.size[0] != max_w:
            new_h = int(p.size[1] * max_w / p.size[0])
            resized_patches.append(p.resize((max_w, new_h), Image.LANCZOS))
        else:
            resized_patches.append(p)

    total_h = sum(p.size[1] for p in resized_patches)
    canvas = Image.new('RGB', (max_w, total_h), (128, 128, 128))

    y_offset = 0
    for p in resized_patches:
        canvas.paste(p, (0, y_offset))
        y_offset += p.size[1]

    return canvas


def _arrange_patches_in_grid(nodes, patches, root_grid_bbox):
    if len(patches) == 1:
        return patches[0]

    patch_grid_positions = []
    for node in nodes:
        if hasattr(node, 'full_path') and len(node.full_path) >= 2:
            second_level_pos = node.full_path[1]
            if second_level_pos in POS_TO_GRID:
                row, col = POS_TO_GRID[second_level_pos]
            else:
                row, col = 1, 1
        else:
            rx1, ry1, rx2, ry2 = root_grid_bbox
            root_w = rx2 - rx1
            root_h = ry2 - ry1
            cell_w = root_w / 3
            cell_h = root_h / 3

            x1, y1, x2, y2 = node.bbox
            cx = (x1 + x2) / 2 - rx1
            cy = (y1 + y2) / 2 - ry1
            col = min(max(int(cx / cell_w), 0), 2)
            row = min(max(int(cy / cell_h), 0), 2)

        patch_grid_positions.append((row, col))

    used_rows = sorted(set(pos[0] for pos in patch_grid_positions))
    used_cols = sorted(set(pos[1] for pos in patch_grid_positions))

    if len(used_rows) == 0 or len(used_cols) == 0:
        return _simple_concat_patches(patches)

    row_map = {r: i for i, r in enumerate(used_rows)}
    col_map = {c: i for i, c in enumerate(used_cols)}

    grid = [[[] for _ in range(len(used_cols))] for _ in range(len(used_rows))]

    for i, (row, col) in enumerate(patch_grid_positions):
        new_row = row_map[row]
        new_col = col_map[col]
        grid[new_row][new_col].append(patches[i])

    merged_grid = [[None] * len(used_cols) for _ in range(len(used_rows))]
    for r_idx in range(len(used_rows)):
        for c_idx in range(len(used_cols)):
            cell_patches = grid[r_idx][c_idx]
            if len(cell_patches) == 0:
                merged_grid[r_idx][c_idx] = None
            elif len(cell_patches) == 1:
                merged_grid[r_idx][c_idx] = cell_patches[0]
            else:
                merged_grid[r_idx][c_idx] = _vertical_concat_patches(cell_patches)

    row_heights = []
    for r_idx in range(len(used_rows)):
        max_h = 0
        for c_idx in range(len(used_cols)):
            if merged_grid[r_idx][c_idx] is not None:
                max_h = max(max_h, merged_grid[r_idx][c_idx].size[1])
        row_heights.append(max_h if max_h > 0 else 100)

    col_widths = []
    for c_idx in range(len(used_cols)):
        max_w = 0
        for r_idx in range(len(used_rows)):
            if merged_grid[r_idx][c_idx] is not None:
                max_w = max(max_w, merged_grid[r_idx][c_idx].size[0])
        col_widths.append(max_w if max_w > 0 else 100)

    canvas_w = sum(col_widths)
    canvas_h = sum(row_heights)
    canvas = Image.new('RGB', (canvas_w, canvas_h), (128, 128, 128))

    y_offset = 0
    for r_idx in range(len(used_rows)):
        x_offset = 0
        for c_idx in range(len(used_cols)):
            patch = merged_grid[r_idx][c_idx]
            if patch is not None:
                target_w = col_widths[c_idx]
                target_h = row_heights[r_idx]
                resized = patch.resize((target_w, target_h), Image.LANCZOS)
                canvas.paste(resized, (x_offset, y_offset))
            x_offset += col_widths[c_idx]
        y_offset += row_heights[r_idx]

    return canvas


def place_crops_by_grid_layout(original_image, selected_nodes):
    """Arrange selected nodes by their root grid positions."""
    if len(selected_nodes) == 0:
        return original_image

    grid_groups = {}
    for node in selected_nodes:
        root_pos = node.root_grid_position or 'middle-center'
        if root_pos not in grid_groups:
            grid_groups[root_pos] = []
        grid_groups[root_pos].append(node)

    grid_images = {}
    for pos, nodes in grid_groups.items():
        patches = []
        for n in nodes:
            if n.eroded_image is not None:
                patches.append(n.eroded_image)
            else:
                patches.append(crop_image(original_image, n.bbox))

        root_grid_bbox = _get_root_grid_bbox(original_image, pos)

        if len(patches) == 1:
            grid_images[pos] = patches[0]
        else:
            grid_images[pos] = _arrange_patches_in_grid(nodes, patches, root_grid_bbox)

    max_w = max(img.size[0] for img in grid_images.values())
    max_h = max(img.size[1] for img in grid_images.values())

    for pos in grid_images:
        if grid_images[pos].size != (max_w, max_h):
            grid_images[pos] = grid_images[pos].resize((max_w, max_h), Image.LANCZOS)

    num_rows = 3
    num_cols = 3

    canvas_w = num_cols * max_w
    canvas_h = num_rows * max_h
    canvas = Image.new('RGB', (canvas_w, canvas_h), (128, 128, 128))

    for pos, img in grid_images.items():
        if pos in POS_TO_GRID:
            row, col = POS_TO_GRID[pos]
            canvas.paste(img, (col * max_w, row * max_h))

    return canvas