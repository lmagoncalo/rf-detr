#!/usr/bin/env python3
"""
YOLO to COCO Dataset Converter

This script converts YOLO format datasets to COCO format manually using standard libraries.
Supports object detection tasks with bounding box annotations.
"""

import argparse
import json
import shutil
from pathlib import Path
import logging
from datetime import datetime
from typing import List, Tuple, Optional
import yaml
from tqdm import tqdm

try:
    import cv2
except ImportError:
    print("Error: OpenCV library not found. Install it with: pip install opencv-python")
    exit(1)

try:
    import yaml
except ImportError:
    print("Error: PyYAML library not found. Install it with: pip install PyYAML")
    exit(1)


class YOLOtoCOCOConverter:
    """Converts YOLO format dataset to COCO format."""

    def __init__(self, yolo_path: Path, output_path: Path, dataset_name: str = "converted_dataset", add_background: bool = True, include_segmentation: bool = False, include_keypoints: bool = False):
        self.yolo_path = yolo_path
        self.output_path = output_path
        self.dataset_name = dataset_name
        self.add_background = add_background
        self.include_segmentation = include_segmentation
        self.include_keypoints = include_keypoints
        self.image_id = 1
        self.annotation_id = 1
        self.categories = []
        self.images = []
        self.annotations = []

    def load_class_names(self) -> List[str]:
        """Load class names from YAML config files or fallback to text files."""
        # First try to load from YAML files
        yaml_classes = self._load_classes_from_yaml()
        if yaml_classes:
            return yaml_classes

        # Fallback to text files
        logging.error("No YAML config found, trying text files...")
        raise ValueError("No valid class names found in YAML or text files. Please ensure your dataset has a proper configuration file.")

    def _load_classes_from_yaml(self) -> List[str]:
        """Load class names from YAML configuration files."""
        yaml_files = ['data.yaml', 'dataset.yaml']

        # Check for specific YAML files first
        for yaml_file in yaml_files:
            yaml_path = self.yolo_path / yaml_file
            if yaml_path.exists():
                logging.info(f"Found YAML config file: {yaml_path}")
                classes = self._parse_yaml_config(yaml_path)
                if classes:
                    return classes

        # If no specific files found, search for any .yaml file
        yaml_files_found = list(self.yolo_path.glob('*.yaml')) + list(self.yolo_path.glob('*.yml'))

        if yaml_files_found:
            logging.info(f"Found YAML files: {[f.name for f in yaml_files_found]}")
            for yaml_path in yaml_files_found:
                logging.info(f"Trying YAML config file: {yaml_path}")
                classes = self._parse_yaml_config(yaml_path)
                if classes:
                    return classes

        return []

    def _parse_yaml_config(self, yaml_path: Path) -> List[str]:
        """Parse YAML config file and extract class names."""
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            if not config:
                logging.warning(f"Empty or invalid YAML file: {yaml_path}")
                return []

            # Look for class names in various formats
            classes = None

            # Check for 'names' key (most common)
            if 'names' in config:
                classes = config['names']
                logging.info(f"Found 'names' in YAML config")

            # Check for 'class_names' key
            elif 'class_names' in config:
                classes = config['class_names']
                logging.info(f"Found 'class_names' in YAML config")

            # Check for 'classes' key
            elif 'classes' in config:
                classes = config['classes']
                logging.info(f"Found 'classes' in YAML config")

            if classes:
                # Handle different formats
                if isinstance(classes, list):
                    # List format: ['class1', 'class2', ...]
                    class_names = [str(name).strip() for name in classes if name]
                elif isinstance(classes, dict):
                    # Dict format: {0: 'class1', 1: 'class2', ...}
                    class_names = []
                    max_id = max(classes.keys()) if classes.keys() else -1
                    for i in range(max_id + 1):
                        if i in classes:
                            class_names.append(str(classes[i]).strip())
                        else:
                            class_names.append(f"class_{i}")
                else:
                    logging.warning(f"Unsupported class names format in {yaml_path}")
                    return []

                # Validate number of classes
                if 'nc' in config:
                    expected_nc = config['nc']
                    if len(class_names) != expected_nc:
                        logging.warning(f"Number of classes ({len(class_names)}) doesn't match 'nc' value ({expected_nc})")

                logging.info(f"Loaded {len(class_names)} classes from YAML: {class_names}")
                return class_names

            else:
                logging.warning(f"No class names found in YAML file: {yaml_path}")
                return []

        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file {yaml_path}: {e}")
            return []
        except Exception as e:
            logging.error(f"Error reading YAML file {yaml_path}: {e}")
            return []

    def find_image_files(self) -> List[Path]:
        """Find all image files based on split file."""
        split_file = self.yolo_path / f"{self.split}.txt"

        image_files = []
        if split_file.exists():
            logging.info(f"Found split file: {split_file}")
            with open(split_file, 'r') as f:
                for line in f:
                    image_path = Path(line.strip())
                    if image_path.exists():
                        image_files.append(image_path)
                    else:
                        logging.warning(f"Image file listed in split file not found: {image_path}")
        else:
            logging.error(f"Split file does not exist: {split_file}")
            return []

        logging.info(f"Found {len(image_files)} image files in {self.split} split")
        return sorted(image_files)

    def find_label_file(self, image_name: str) -> Optional[Path]:
        """Find corresponding label file for an image."""
        # Get the split directory
        labels_path = self.yolo_path / 'labels'
        labels_dir = labels_path / self.split

        if labels_dir.exists():
            label_path = labels_dir / f"{Path(image_name).stem}.txt"
            if label_path.exists():
                return label_path

        return None

    def yolo_to_coco_bbox(self, yolo_bbox: List[float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Convert YOLO bbox format to COCO format."""
        x_center, y_center, width, height = yolo_bbox

        # Convert normalized coordinates to pixel coordinates
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        # Convert to COCO format (top-left x, top-left y, width, height)
        x = x_center - width / 2
        y = y_center - height / 2

        return x, y, width, height
    
    def yolo_to_coco_segmentation(self, yolo_segmentation: List[float], img_width: int, img_height: int) -> Tuple[List[float], float, float, float, float]:
        """Convert YOLO bbox format to COCO format."""
        points = yolo_segmentation

        for i in range(0, len(points), 2):
            points[i] *= img_width
            points[i + 1] *= img_height

        # Calculate bounding box from segmentation points
        x_coords = points[0::2]
        y_coords = points[1::2]
        x = min(x_coords)
        y = min(y_coords)
        width = max(x_coords) - x
        height = max(y_coords) - y

        return points, x, y, width, height
    
    def yolo_to_coco_keypoints(self, yolo_keypoints: List[float], img_width: int, img_height: int) -> Tuple[List[float], float, float, float, float]:
        """Convert YOLO bbox format to COCO format."""
        x_center, y_center, width, height, x1, y1, conf1, x2, y2, conf2 = yolo_keypoints

        # Scale to image dimensions
        x_center, width, x1, x2 = [v * img_width  for v in [x_center, width, x1, x2]]
        y_center, height, y1, y2 = [v * img_height for v in [y_center, height, y1, y2]]

        # Convert to COCO format (top-left x, top-left y, width, height)
        return [x1, y1, int(conf1), x2, y2, int(conf2)], x_center - width / 2, y_center - height / 2, width, height

    def process_image(self, image_path: Path, class_names: List[str]) -> bool:
        """Process a single image and its annotations."""
        try:
            # Load image to get dimensions using OpenCV
            img = cv2.imread(str(image_path))
            if img is None:
                logging.error(f"Could not load image: {image_path}")
                return False

            img_height, img_width = img.shape[:2]

            # Create image entry
            image_info = {
                "id": self.image_id,
                "file_name": image_path.name,
                "width": img_width,
                "height": img_height,
                "date_captured": datetime.now().isoformat()
            }
            self.images.append(image_info)

            # Find and process label file
            label_path = self.find_label_file(image_path.name)
            if label_path:
                self.process_annotations(label_path, self.image_id, img_width, img_height)
            else:
                logging.warning(f"No label file found for {image_path.name}")

            self.image_id += 1
            return True

        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            return False

    def process_annotations(self, label_path: Path, image_id: int, img_width: int, img_height: int):
        """Process annotations from a YOLO label file."""
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()

                    if self.include_segmentation:
                        if (len(parts) - 1) % 2 == 0:
                            class_id = int(parts[0])
                            parts = list(map(float, parts[1:]))

                            # Convert YOLO bbox to COCO format
                            coords, x, y, w, h = self.yolo_to_coco_segmentation(parts, img_width, img_height)

                            # Create annotation entry with segmentation
                            annotation = {
                                "id": self.annotation_id,
                                "image_id": image_id,
                                "category_id": class_id + 1,
                                "bbox": [x, y, w, h],
                                "area": w * h,
                                "iscrowd": 0,
                                "segmentation": [coords]  # Assuming the rest are segmentation points
                            }

                            self.annotations.append(annotation)
                            self.annotation_id += 1
                        else:
                            logging.error(f"Error processing segmentation annotations from {label_path}: Size is not multiple of 2: {len(parts)}")
                    elif self.include_keypoints:
                        if (len(parts) - 5) % 3 == 0:
                            class_id = int(parts[0])
                            parts = list(map(float, parts[1:]))

                            # Convert YOLO bbox to COCO format
                            coords, x, y, w, h = self.yolo_to_coco_keypoints(parts, img_width, img_height)

                            # Create annotation entry with keypoints
                            annotation = {
                                "id": self.annotation_id,
                                "image_id": image_id,
                                "category_id": class_id + 1,
                                "bbox": [x, y, w, h],
                                "area": w * h,
                                "iscrowd": 0,
                                "num_keypoints": 2,
                                "segmentation": [],
                                "keypoints": coords  # Assuming the rest are keypoints points
                            }

                            self.annotations.append(annotation)
                            self.annotation_id += 1
                        else:
                            logging.error(f"Error processing keypoints annotations from {label_path}: Size is not multiple of 3: {len(parts)}")
                    else:
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])

                            # Convert YOLO bbox to COCO format
                            x, y, w, h = self.yolo_to_coco_bbox([x_center, y_center, width, height], img_width, img_height)

                            # Create annotation entry
                            annotation = {
                                "id": self.annotation_id,
                                "image_id": image_id,
                                "category_id": class_id + 1,
                                "bbox": [x, y, w, h],
                                "area": w * h,
                                "iscrowd": 0,
                                "segmentation": []
                            }

                            self.annotations.append(annotation)
                            self.annotation_id += 1
                        
        except Exception as e:
            logging.error(f"Error processing annotations from {label_path}: {e}")

    def create_categories(self, class_names: List[str]):
        """Create COCO categories from class names."""
        self.categories = []

        # Add background class if requested
        if self.add_background:
            background_category = {
                "id": 0,
                "name": "background",
                "supercategory": "none"
            }
            self.categories.append(background_category)

        # Add regular classes
        for i, class_name in enumerate(class_names):
            category = {
                "id": i + 1,
                "name": class_name,
                "supercategory": "object"
            }
            self.categories.append(category)

    def save_coco_dataset(self):
        """Save the COCO format dataset."""
        # Create output directory for this split
        if self.split == "val":
            split = "valid"
        else:
            split = self.split

        split_output_dir = self.output_path / split
        split_output_dir.mkdir(parents=True, exist_ok=True)

        # Create COCO JSON structure
        coco_data = {
            "info": {
                "description": f"{self.dataset_name} - {self.split} split",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "YOLO to COCO Converter",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown",
                    "url": ""
                }
            ],
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories
        }

        # Save annotations with COCO naming convention
        annotations_file = split_output_dir / '_annotations.coco.json'
        with open(annotations_file, 'w') as f:
            json.dump(coco_data, f, indent=2)

        logging.info(f"Saved annotations to: {annotations_file}")

        # Copy images to split directory
        logging.info(f"Copying images to {self.split} directory...")
        image_files = self.find_image_files()
        copied_count = 0

        for image_path in image_files:
            dst_path = split_output_dir / image_path.name
            try:
                shutil.copy2(image_path, dst_path)
                copied_count += 1
            except Exception as e:
                logging.error(f"Error copying {image_path}: {e}")

        logging.info(f"Copied {copied_count} images to: {split_output_dir}")

    def convert(self):
        """Main conversion method."""
        logging.info(f"Starting conversion from {self.yolo_path} to {self.output_path}")

        if self.add_background:
            logging.info("Background class will be added to categories")
        else:
            logging.info("Background class will NOT be added to categories")

        # Load class names
        class_names = self.load_class_names()
        if not class_names:
            raise ValueError("Could not load or infer class names")

        # Create categories
        self.create_categories(class_names)

        # Process all images
        for split in ["train", "val"]:
            # Reset data for each split
            self.split = split
            self.annotations = []
            self.images = []
            self.image_id = 1
            self.annotation_id = 1

            logging.info(f"Processing split: {self.split}")

            image_files = self.find_image_files()
            if not image_files:
                raise ValueError("No image files found in the dataset")

            processed_count = 0
            for image_path in tqdm(image_files):
                if self.process_image(image_path, class_names):
                    processed_count += 1

            logging.info(f"Processed {processed_count} images with {len(self.annotations)} annotations")

            # Save COCO dataset
            self.save_coco_dataset()

            # Print summary
            logging.info(f"Conversion Summary Split {self.split}:")
            logging.info(f"  - Images: {len(self.images)}")
            logging.info(f"  - Annotations: {len(self.annotations)}")
            logging.info(f"  - Categories: {len(self.categories)}")
            if self.add_background:
                logging.info(f"  - Background class: Added (id: 0)")
            logging.info(f"  - Output: {self.output_path}")

        # Create a test split equal to the validation split
        logging.info("Creating test split equal to validation split...")
        
        # Copy validation folder and call it test
        val_dir = self.output_path / "valid"
        test_dir = self.output_path / "test"
        if val_dir.exists():
            if test_dir.exists():
                logging.warning(f"Test directory already exists and will be overwritten: {test_dir}")
                shutil.rmtree(test_dir)
            shutil.copytree(val_dir, test_dir)
            logging.info(f"Test split created successfully at: {test_dir}")


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main function to handle CLI arguments and orchestrate conversion."""
    parser = argparse.ArgumentParser(
        description="Convert YOLO dataset to COCO format manually",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input /path/to/yolo/dataset --output /path/to/coco/output
  %(prog)s -i ./yolo_data -o ./coco_data --name "my_dataset" --verbose
  %(prog)s --input ./yolo_dataset --output ./coco_train --split train --force
  %(prog)s --input ./yolo_data --output ./coco_data --no-background

YOLO Dataset Structure:
  Your YOLO dataset should have this structure:

  yolo_dataset/
  ├── train/
  │   ├── images/
  │   │   ├── img1.jpg
  │   │   └── img2.jpg
  │   └── labels/
  │       ├── img1.txt
  │       └── img2.txt
  ├── valid/
  │   ├── images/
  │   └── labels/
  ├── test/
  │   ├── images/
  │   └── labels/
  ├── data.yaml (or dataset.yaml)
  └── classes.txt (optional, if no YAML)
        """
    )

    parser.add_argument(
        '-i', '--input',
        type=Path,
        required=True,
        help='Path to input YOLO dataset directory'
    )

    parser.add_argument(
        '-o', '--output',
        type=Path,
        required=True,
        help='Path to output directory for COCO dataset'
    )

    parser.add_argument(
        '-n', '--name',
        type=str,
        default='converted_dataset',
        help='Name for the converted dataset (default: converted_dataset)'
    )

    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Force overwrite output directory if it exists'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--segmentation',
        action='store_true',
        help='Include segmentation masks in the output (if available in YOLO format)'
    )

    parser.add_argument(
        '--keypoints',
        action='store_true',
        help='Include keypoints in the output (if available in YOLO format)'
    )

    parser.add_argument(
        '--no-background',
        action='store_true',
        help='Do not add background class (id: 0) to categories'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Check output path
    if args.output.exists():
        if not args.force:
            logging.error(f"Output directory already exists: {args.output}")
            logging.error("Use --force to overwrite or choose a different output path")
            exit(1)
        else:
            logging.warning(f"Output directory exists and will be overwritten: {args.output}")

    # Determine whether to add background class
    add_background = not args.no_background

    try:
        # Convert specific split
        logging.info(f"Converting images...")
        converter = YOLOtoCOCOConverter(args.input, args.output, args.name, add_background, args.segmentation, args.keypoints)
        converter.convert()
        logging.info("Conversion completed successfully!")

    except KeyboardInterrupt:
        logging.info("Conversion interrupted by user")
        exit(1)
    except Exception as e:
        logging.error(f"Conversion failed: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
