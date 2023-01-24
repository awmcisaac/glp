# Adapted from https://github.com/kareem-metwaly/glidenet/blob/master/dataset/vaw/dataset.py

import os
import typing as t
from dataclasses import dataclass

from PIL import Image
import pandas as pd
import json
import torch
import torchvision.transforms as T
import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import ToPILImage


@dataclass
class VAWDataSample:
    image_id: int  # corresponds to the same id in VG dataset
    image_path: t.Optional[str]  # attempt to read the image out of that path
    instance_id: int  # Unique instance ID
    instance_bbox: t.Tuple[float, float, float, float]  # [left, upper, width, height]
#    instance_polygon: t.List[
#        t.List[t.Tuple[float, float]]
#    ]  # number of polygons, nested is list of (x,y) coordinates of each polygon
    object_name: str  # Name of the object for the instance
    positive_attributes: t.List[str]  # Explicitly labeled positive attributes
    negative_attributes: t.List[str]  # Explicitly labeled negative attributes

    @staticmethod
    def from_dict(sample: t.Dict[str, t.Any], path: os.path) -> "Downsample":
        image_id = int(sample["image_id"])
        data_sample = VAWDataSample(
            image_id=image_id,
            image_path=os.path.join(path, str(image_id) + ".jpg"),
            instance_id=int(sample["instance_id"]),
            instance_bbox=sample["instance_bbox"],
            object_name=sample["object_name"],
            positive_attributes=sample["positive_attributes"],
            negative_attributes=sample["negative_attributes"],
        )

        return data_sample

    @property
    def retrieve_image(self) -> Image.Image:
        return Image.open(self.image_path).convert("RGB")

    def crop_image(self, image) -> Image.Image:
        l, u, w, h = self.instance_bbox
        return image.crop(box=(l,u,l+w,u+h)) # [left, upper, right, lower]


class VAWDataset(Dataset):
    def __init__(self,
                 annotations_file: os.path,
                 attrib_idx_file: os.path,
                 img_dir: os.path,
                 split: t.Literal["train", "val", "test"],
                 normalize: bool = True,
                 resize: t.Optional[int] = None,
                 device="cuda:0"):
        self.img_dir = img_dir
        with open(annotations_file, "r") as f:
            vaw_data = json.load(f)
        self.data = [VAWDataSample.from_dict(sample, self.img_dir) 
            for sample in vaw_data]
        self.data = [
            sample for sample in self.data
            if len(sample.positive_attributes + sample.negative_attributes) > 0]

        with open(attrib_idx_file, "r") as f:
            self.att2idx = json.load(f)
        self.idx2att = {val: key for (key, val) in self.att2idx.items()}
        self.objects = sorted({sample.object_name for sample in self.data})
        self.obj2idx = {key: val for (key, val) in 
            zip(self.objects, range(len(self.objects)))}
        self.idx2obj = {val: key for (key, val) in 
            zip(self.objects, range(len(self.objects)))}

        self.split = split
        self.normalize = normalize
        self.resize = resize

        self.augmentations = []
        self.augmentations.append(T.ToTensor())
        if self.normalize:
            # Normalized with mean and std from https://pytorch.org/vision/0.8/models.html
            self.augmentations.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225]))
        if self.resize:
            self.augmentations.append(T.Resize(size=(self.resize, self.resize)))
        self.augmentations = T.Compose(self.augmentations)

    def encode_attrs(self, pos_atts: t.List[str], neg_atts: t.List[str]) -> torch.Tensor:
        """
        Encode attribute labels with 1 for positive, 
        0 for negative, -1 for missing
        """
        encoding = torch.neg(torch.ones(len(self.att2idx)))
        for pos_att in pos_atts:
            encoding[self.att2idx[pos_att]] = 1
        for neg_att in neg_atts:
            encoding[self.att2idx[neg_att]] = 0
        return encoding

    def decode_attrs(self, encoding: torch.Tensor) -> t.Dict[str, t.List[str]]:
        """
        Decode Tensor into positive and negative attribute label lists
        """
        pos_atts = [self.idx2att[int(idx)] 
            for idx in (encoding==1).nonzero(as_tuple=True)[0]
        ]
        neg_atts = [self.idx2att[int(idx)] 
            for idx in (encoding==0).nonzero(as_tuple=True)[0]
        ]
        return {"positive_attributes": pos_atts, "negative_attributes": neg_atts}

    def encode_objs(self, object_name: str) -> torch.Tensor:
        return torch.tensor(self.obj2idx[object_name])

    def decode_objs(self, idx: torch.Tensor) -> str:
        return self.idx2obj[int(idx)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> t.Dict[str, t.Any]:
        sample = self.data[idx]
        image = sample.retrieve_image
        attr_label = self.encode_attrs(
            sample.positive_attributes, sample.negative_attributes
        )
        instance_id = sample.instance_id
        instance_bbox = sample.instance_bbox
        object_name = sample.object_name
        object_id = self.encode_objs(object_name)
        cropped_image = sample.crop_image(image)
        if self.augmentations:
            image = self.augmentations(image)
            cropped_image = self.augmentations(cropped_image)

        output = {"image": image,
                  "cropped_image": cropped_image,
                  "object_name": object_name,
                  "object_id": object_id,
                  "instance_id": instance_id,
                  "attributes_label": attr_label,
                  "id": torch.tensor(idx)
                 }
        return output

    @property
    def num_objects(self):
        return len(self.obj2idx)

    @property
    def num_categories(self):
        return len(self.att2idx)

if __name__ == "__main__":
    dataset = VAWDataset(annotations_file="data/test.json",
                         attrib_idx_file="data/attribute_index.json",
                         img_dir="VG_100K",
                         split="test",
                         normalize=True,
                         resize=224)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        num_workers=0,
        pin_memory=True,
    )
    for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        if batch["object_name"][0] == "floor":
            print(dataset.decode_objs(batch["object_id"][0]))
            print(batch["image"].shape)
            ToPILImage()(batch["image"][0]).show()
            continue
        pass
