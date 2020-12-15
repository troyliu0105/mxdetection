import argparse
import os

import tqdm
from pycocotools import coco

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str)
    parser.add_argument('--root', type=str)
    parser.add_argument('--out', type=str)
    opts = parser.parse_args()
    annos = coco.COCO(opts.json)
    cat2id = {k: i for i, k in enumerate(list(annos.cats.keys()))}

    imgIds = annos.getImgIds()
    lines = []
    for idx, img_id in tqdm.tqdm(enumerate(imgIds), desc='Converting...', total=len(imgIds)):
        anno_ids = annos.getAnnIds(imgIds=img_id)
        img_info = annos.loadImgs(img_id)[0]
        height = img_info['height']
        width = img_info['width']
        filename = img_info['file_name']
        line = [f"{idx}", "4", "6", f"{float(width)}", f"{float(height)}"]
        for anno in annos.loadAnns(anno_ids):
            iscrowd = anno['iscrowd']
            bbox = anno['bbox']
            category_id = anno['category_id']
            class_id = cat2id[category_id]
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            bbox[0] /= width
            bbox[1] /= height
            bbox[2] /= width
            bbox[3] /= height
            bbox.insert(0, class_id)
            bbox.append(iscrowd)
            line.append("\t".join([str(float(b)) for b in bbox]))
        line.append(os.path.join(opts.root, filename))

        lines.append("\t".join(line))
    with open(opts.out, 'w') as fp:
        fp.writelines([l + '\n' for l in lines])
