import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm


DEFAULT_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Convert VOC XML annotations to YOLO txt labels.")
    parser.add_argument(
        "--voc-root",
        type=Path,
        required=True,
        help="VOC dataset root, e.g. /path/to/VOC2007",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root for YOLO-format dataset.",
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["train", "val"],
        help="Image set names under ImageSets/Main, e.g. train val test trainval.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="Class names in label order.",
    )
    parser.add_argument(
        "--classes-file",
        type=Path,
        default=None,
        help="Optional text file with one class name per line. Overrides --classes.",
    )
    parser.add_argument(
        "--image-ext",
        default=".jpg",
        help="Image suffix used in JPEGImages, e.g. .jpg or .png.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into output_root/images/<set>/.",
    )
    parser.add_argument(
        "--skip-difficult",
        action="store_true",
        help="Skip objects whose <difficult> is 1.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep empty label files for images without valid objects.",
    )
    return parser.parse_args()


def load_classes(args):
    if args.classes_file is None:
        return args.classes

    class_names = []
    with args.classes_file.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                class_names.append(name)
    if not class_names:
        raise ValueError(f"No classes found in {args.classes_file}")
    return class_names


def convert_box(size, box):
    width, height = size
    xmin, xmax, ymin, ymax = box
    x_center = ((xmin + xmax) / 2.0 - 1.0) / width
    y_center = ((ymin + ymax) / 2.0 - 1.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height


def read_size(root, xml_path):
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing <size> in {xml_path}")

    width = int(float(size.findtext("width", default="0")))
    height = int(float(size.findtext("height", default="0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}: width={width}, height={height}")
    return width, height


def build_label_lines(xml_path, class_to_id, skip_difficult):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    width, height = read_size(root, xml_path)
    label_lines = []
    skipped_classes = []

    for obj in root.iter("object"):
        class_name = obj.findtext("name", default="").strip()
        difficult = int(obj.findtext("difficult", default="0"))
        if skip_difficult and difficult == 1:
            continue
        if class_name not in class_to_id:
            skipped_classes.append(class_name)
            continue

        bbox = obj.find("bndbox")
        if bbox is None:
            continue

        xmin = float(bbox.findtext("xmin", default="0"))
        xmax = float(bbox.findtext("xmax", default="0"))
        ymin = float(bbox.findtext("ymin", default="0"))
        ymax = float(bbox.findtext("ymax", default="0"))
        if xmax <= xmin or ymax <= ymin:
            continue

        x_center, y_center, box_width, box_height = convert_box((width, height), (xmin, xmax, ymin, ymax))
        class_id = class_to_id[class_name]
        label_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )

    return label_lines, skipped_classes


def convert_set(voc_root, output_root, set_name, class_to_id, image_ext, copy_images, skip_difficult, keep_empty):
    set_file = voc_root / "ImageSets" / "Main" / f"{set_name}.txt"
    if not set_file.exists():
        raise FileNotFoundError(f"Set file not found: {set_file}")

    image_ids = [line.strip() for line in set_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    label_dir = output_root / "labels" / set_name
    image_dir = output_root / "images" / set_name
    label_dir.mkdir(parents=True, exist_ok=True)
    if copy_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    empty = 0
    missing_xml = []
    missing_images = []
    unknown_classes = set()

    for image_id in tqdm(image_ids, desc=f"convert {set_name}"):
        xml_path = voc_root / "Annotations" / f"{image_id}.xml"
        img_path = voc_root / "JPEGImages" / f"{image_id}{image_ext}"
        if not xml_path.exists():
            missing_xml.append(str(xml_path))
            continue
        if copy_images and not img_path.exists():
            missing_images.append(str(img_path))
            continue

        label_lines, skipped_classes = build_label_lines(xml_path, class_to_id, skip_difficult)
        unknown_classes.update(skipped_classes)

        label_path = label_dir / f"{image_id}.txt"
        if label_lines or keep_empty:
            label_path.write_text("\n".join(label_lines), encoding="utf-8")
            converted += 1
        else:
            empty += 1

        if copy_images:
            shutil.copy2(img_path, image_dir / img_path.name)

    return {
        "set_name": set_name,
        "images_total": len(image_ids),
        "converted": converted,
        "empty": empty,
        "missing_xml": missing_xml,
        "missing_images": missing_images,
        "unknown_classes": sorted(x for x in unknown_classes if x),
    }


def main():
    args = parse_args()
    class_names = load_classes(args)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    summaries = []
    for set_name in args.sets:
        summary = convert_set(
            voc_root=args.voc_root,
            output_root=args.output_root,
            set_name=set_name,
            class_to_id=class_to_id,
            image_ext=args.image_ext,
            copy_images=args.copy_images,
            skip_difficult=args.skip_difficult,
            keep_empty=args.keep_empty,
        )
        summaries.append(summary)

    print("Classes:", class_names)
    for summary in summaries:
        print(
            f"[{summary['set_name']}] total={summary['images_total']} "
            f"converted={summary['converted']} empty_skipped={summary['empty']} "
            f"missing_xml={len(summary['missing_xml'])} missing_images={len(summary['missing_images'])}"
        )
        if summary["unknown_classes"]:
            print(f"  unknown classes skipped: {summary['unknown_classes']}")


if __name__ == "__main__":
    main()
