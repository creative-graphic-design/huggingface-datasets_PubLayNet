import json
import pathlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import datasets as ds
import numpy as np
from datasets.utils.logging import get_logger
from PIL import Image
from PIL.Image import Image as PilImage
from pycocotools import mask as cocomask
from tqdm.auto import tqdm

logger = get_logger(__name__)

JsonDict = Dict[str, Any]
ImageId = int
Bbox = Tuple[float, float, float, float]
AnnotationId = int
LicenseId = int
CategoryId = int
Bbox = Tuple[float, float, float, float]

_DESCRIPTION = ""

_CITATION = ""

_HOMEPAGE = ""

_LICENSE = ""

_URL = "https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/publaynet.tar.gz"


class UncompressedRLE(TypedDict):
    counts: List[int]
    size: Tuple[int, int]


class CompressedRLE(TypedDict):
    counts: bytes
    size: Tuple[int, int]


@dataclass
class CategoryData(object):
    category_id: int
    name: str
    supercategory: str

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "CategoryData":
        return cls(
            category_id=json_dict["id"],
            name=json_dict["name"],
            supercategory=json_dict["supercategory"],
        )


@dataclass
class ImageData(object):
    image_id: ImageId
    file_name: str
    width: int
    height: int

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "ImageData":
        return cls(
            image_id=json_dict["id"],
            file_name=json_dict["file_name"],
            width=json_dict["width"],
            height=json_dict["height"],
        )

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)


@dataclass
class AnnotationData(object):
    annotation_id: AnnotationId
    image_id: ImageId
    segmentation: Union[np.ndarray, CompressedRLE]
    area: float
    iscrowd: bool
    bbox: Bbox
    category_id: int

    @classmethod
    def compress_rle(
        cls,
        segmentation: Union[List[List[float]], UncompressedRLE],
        iscrowd: bool,
        height: int,
        width: int,
    ) -> CompressedRLE:
        if iscrowd:
            rle = cocomask.frPyObjects(segmentation, h=height, w=width)
        else:
            rles = cocomask.frPyObjects(segmentation, h=height, w=width)
            rle = cocomask.merge(rles)

        return rle  # type: ignore

    @classmethod
    def rle_segmentation_to_binary_mask(
        cls, segmentation, iscrowd: bool, height: int, width: int
    ) -> np.ndarray:
        rle = cls.compress_rle(
            segmentation=segmentation, iscrowd=iscrowd, height=height, width=width
        )
        return cocomask.decode(rle)  # type: ignore

    @classmethod
    def rle_segmentation_to_mask(
        cls,
        segmentation: Union[List[List[float]], UncompressedRLE],
        iscrowd: bool,
        height: int,
        width: int,
    ) -> np.ndarray:
        binary_mask = cls.rle_segmentation_to_binary_mask(
            segmentation=segmentation, iscrowd=iscrowd, height=height, width=width
        )
        return binary_mask * 255

    @classmethod
    def from_dict(
        cls,
        json_dict: JsonDict,
        images: Dict[ImageId, ImageData],
        decode_rle: bool,
    ) -> "AnnotationData":
        segmentation = json_dict["segmentation"]
        image_id = json_dict["image_id"]
        image_data = images[image_id]
        iscrowd = bool(json_dict["iscrowd"])

        segmentation_mask = (
            cls.rle_segmentation_to_mask(
                segmentation=segmentation,
                iscrowd=iscrowd,
                height=image_data.height,
                width=image_data.width,
            )
            if decode_rle
            else cls.compress_rle(
                segmentation=segmentation,
                iscrowd=iscrowd,
                height=image_data.height,
                width=image_data.width,
            )
        )
        return cls(
            annotation_id=json_dict["id"],
            image_id=image_id,
            segmentation=segmentation_mask,
            area=json_dict["area"],
            iscrowd=iscrowd,
            bbox=json_dict["bbox"],
            category_id=json_dict["category_id"],
        )


def load_json(json_path: pathlib.Path) -> JsonDict:
    logger.info(f"Load from {json_path}")
    with json_path.open("r") as rf:
        json_dict = json.load(rf)
    return json_dict


def load_image(image_path: pathlib.Path) -> PilImage:
    return Image.open(image_path)


def load_categories_data(
    category_dicts: List[JsonDict],
    tqdm_desc: str = "Load categories",
) -> Dict[CategoryId, CategoryData]:
    categories = {}
    for category_dict in tqdm(category_dicts, desc=tqdm_desc):
        category_data = CategoryData.from_dict(category_dict)
        categories[category_data.category_id] = category_data
    return categories


def load_images_data(
    image_dicts: List[JsonDict],
    tqdm_desc="Load images",
) -> Dict[ImageId, ImageData]:
    images = {}
    for image_dict in tqdm(image_dicts, desc=tqdm_desc):
        image_data = ImageData.from_dict(image_dict)
        images[image_data.image_id] = image_data
    return images


def load_annotation_data(
    label_dicts: List[JsonDict],
    images: Dict[ImageId, ImageData],
    decode_rle: bool,
    tqdm_desc: str = "Load label data",
) -> Dict[ImageId, List[AnnotationData]]:
    labels = defaultdict(list)
    label_dicts = sorted(label_dicts, key=lambda d: d["image_id"])

    for label_dict in tqdm(label_dicts, desc=tqdm_desc):
        label_data = AnnotationData.from_dict(
            label_dict, images=images, decode_rle=decode_rle
        )
        labels[label_data.image_id].append(label_data)
    return labels


def generate_examples(
    annotations: Dict[ImageId, List[AnnotationData]],
    image_dir: pathlib.Path,
    images: Dict[ImageId, ImageData],
    categories: Dict[CategoryId, CategoryData],
):
    for idx, image_id in enumerate(images.keys()):
        image_data = images[image_id]
        image_anns = annotations[image_id]

        if len(image_anns) < 1:
            logger.warning(f"No annotation found for image id: {image_id}.")
            continue

        image = load_image(image_path=image_dir / image_data.file_name)
        example = asdict(image_data)
        example["image"] = image

        example["annotations"] = []
        for ann in image_anns:
            ann_dict = asdict(ann)
            category = categories[ann.category_id]
            ann_dict["category"] = asdict(category)
            example["annotations"].append(ann_dict)

        print(example)
        yield idx, example


@dataclass
class PubLayNetConfig(ds.BuilderConfig):
    decode_rle: bool = False


class PubLayNetDataset(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIG_CLASS = PubLayNetConfig
    BUILDER_CONFIGS = [
        PubLayNetConfig(
            version=VERSION,
            description="TBD",
        )
    ]

    def _info(self) -> ds.DatasetInfo:
        segmentation_feature = (
            ds.Image()
            if self.config.decode_rle
            else {
                "counts": ds.Sequence(ds.Value("binary")),
                "size": ds.Sequence(ds.Value("int32")),
            }
        )
        features = ds.Features(
            {
                "image_id": ds.Value("int32"),
                "file_name": ds.Value("string"),
                "width": ds.Value("int32"),
                "height": ds.Value("int32"),
                "image": ds.Image(),
                "annotations": ds.Sequence(
                    {
                        "annotation_id": ds.Value("int32"),
                        "area": ds.Value("float32"),
                        "bbox": ds.Sequence(ds.Value("float32"), length=4),
                        "category": {
                            "category_id": ds.Value("int32"),
                            "name": ds.ClassLabel(
                                num_classes=5,
                                names=["text", "title", "list", "table", "figure"],
                            ),
                            "supercategory": ds.Value("string"),
                        },
                        "category_id": ds.Value("int32"),
                        "image_id": ds.Value("int32"),
                        "iscrowd": ds.Value("bool"),
                        "segmentation": segmentation_feature,
                    }
                ),
            }
        )
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            features=features,
        )

    def _split_generators(self, dl_manager: ds.DownloadManager):
        base_dir = dl_manager.download_and_extract(_URL)
        publaynet_dir = pathlib.Path(base_dir) / "publaynet"

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs={
                    "image_dir": publaynet_dir / "train",
                    "label_path": publaynet_dir / "train.json",
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,
                gen_kwargs={
                    "image_dir": publaynet_dir / "val",
                    "label_path": publaynet_dir / "val.json",
                },
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,
                gen_kwargs={
                    "image_dir": publaynet_dir / "test",
                },
            ),
        ]

    def _generate_train_val_examples(
        self, image_dir: pathlib.Path, label_path: pathlib.Path
    ):
        label_json = load_json(json_path=label_path)

        images = load_images_data(image_dicts=label_json["images"])
        categories = load_categories_data(category_dicts=label_json["categories"])

        annotations = load_annotation_data(
            label_dicts=label_json["annotations"],
            images=images,
            decode_rle=self.config.decode_rle,
        )
        yield from generate_examples(
            annotations=annotations,
            image_dir=image_dir,
            images=images,
            categories=categories,
        )

    def _generate_examples(
        self, image_dir: pathlib.Path, label_path: Optional[pathlib.Path] = None
    ):
        is_test = label_path is None

        if not is_test:
            yield from self._generate_train_val_examples(
                image_dir=image_dir,
                label_path=label_path,
            )
        else:
            yield from self._generate_test_examples(
                image_dir=image_dir,
            )
