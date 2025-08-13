import json
from pathlib import Path

def convert_coco_json_to_flat_list(
    coco_json_path: str,
    image_root_dir: str,
    save_path: str = None
):
    coco = json.load(open(coco_json_path, 'r'))
    image_id_to_name = {img['id']: img['file_name'] for img in coco['images']}

    flat_list = []
    for ann in coco['annotations']:
        image_id = ann['image_id']
        caption = ann['caption']
        if image_id in image_id_to_name:
            image_name = image_id_to_name[image_id]
            full_path = str(Path(image_root_dir) / image_name)
            flat_list.append({
                'path': full_path,
                'caption': caption
            })

    if save_path:
        Path(save_path).write_text(json.dumps(flat_list, indent=2), encoding='utf-8')
        print(f"✅ 已保存转换后的 JSON，共 {len(flat_list)} 条 → {save_path}")
    return flat_list


convert_coco_json_to_flat_list(
    coco_json_path="/share/project/data/COCO2017/annotations/captions_train2017.json",
    image_root_dir="/share/project/data/COCO2017/train2017",
    save_path="/home/easytaker/data/flat_captions_train2017.json"
)
