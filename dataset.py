# Adapted from https://github.com/kareem-metwaly/glidenet/blob/master/dataset/vaw/dataset.py

import os
import typing as t
from dataclasses import dataclass
from collections import defaultdict

from PIL import Image
import json
import torch
import torchvision.transforms as T
import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage


@dataclass
class VAWDataSample:
    image_id: int  # corresponds to the same id in VG dataset
    image_path: t.Optional[str]  # attempt to read the image out of that path
    instance_id: int  # Unique instance ID
    instance_bbox: t.Tuple[float, float, float, float]  # [left, upper, width, height]
    object_name: str  # Name of the object for the instance
    positive_attributes: t.List[str]  # Explicitly labeled positive attributes
    negative_attributes: t.List[str]  # Explicitly labeled negative attributes

    @staticmethod
    def from_dict(sample: t.Dict[str, t.Any], path: os.path) -> "VAWDataSample":
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
        # for each bounding box in the dataset, we expand its width and
        # height by min(w,h)*0.3 to capture more context.
        expand = min(w, h) * 0.3
        l -= expand / 2
        u -= expand / 2
        w += expand
        h += expand

        return image.crop(box=(l, u, l + w, u + h))  # [left, upper, right, lower]


class VAWDataset(Dataset):
    def __init__(self,
                 annotations_file: os.path,
                 attrib_idx_file: os.path,
                 attrib_parent_types: os.path,  # TODO: Make config file for these
                 attrib_types: os.path,
                 img_dir: os.path,
                 split: t.Literal["train", "val", "test"],
                 resize: int = 224,
                 device="cuda:0"):
        self.img_dir = img_dir
        with open(annotations_file, "r") as f:
            vaw_data = json.load(f)
        self.data = [
            VAWDataSample.from_dict(sample, self.img_dir) for sample in vaw_data
        ]
        self.data = [
            sample for sample in self.data
            if len(sample.positive_attributes + sample.negative_attributes) > 0
        ]

        with open(attrib_idx_file, "r") as f:
            self.att2idx = json.load(f)
        self.idx2att = {val: key for (key, val) in self.att2idx.items()}
        with open(attrib_parent_types, "r") as f:
            self.parent2type = json.load(f)
        with open(attrib_types, "r") as f:
            self.type2attrib = json.load(f)
        self.parent2attrib = defaultdict(list)
        for par, typ in self.parent2type.items():
            for t in typ:
                try:
                    self.parent2attrib[par].extend(self.type2attrib[t])
                except KeyError:
                    pass

        self.objects = sorted({sample.object_name for sample in self.data})
        self.obj2idx = {key: val for (key, val) in
                        zip(self.objects, range(len(self.objects)))}
        self.idx2obj = {val: key for (key, val) in
                        zip(self.objects, range(len(self.objects)))}

        self.split = split
        self.resize = resize

        self.augmentations = []
        self.augmentations.append(T.ToTensor())
        # normalized with mean and std from https://pytorch.org/vision/0.8/models.html
        self.augmentations.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]))
        # random crop around the expanded bounding box
        self.augmentations.append(T.RandomCrop(size=(self.resize, self.resize),
                                               pad_if_needed=True))
        self.augmentations.append(T.ColorJitter(brightness=0.2,
                                                contrast=0.2,
                                                saturation=0.2,
                                                hue=0.2))
        self.augmentations.append(T.RandomHorizontalFlip())
        self.augmentations = T.Compose(self.augmentations)
        self.random_grayscale = T.RandomGrayscale()

    def encode_attrs(self, pos_atts: t.List[str], neg_atts: t.List[str]) -> torch.Tensor:
        """
        Encode attribute labels with 1 for positive, 
        0 for negative, -1 for missing
        """
        encoding = torch.neg(torch.ones(len(self.att2idx)))
        encoding[[self.att2idx[pos_att] for pos_att in pos_atts]] = 1
        encoding[[self.att2idx[neg_att] for neg_att in neg_atts]] = 0
        return encoding

    def decode_attrs(self, encoding: torch.Tensor) -> t.Dict[str, t.List[str]]:
        """
        Decode Tensor into positive and negative attribute label lists
        """
        pos_atts = [self.idx2att[int(idx)]
                    for idx in (encoding == 1).nonzero(as_tuple=True)[0]
                   ]
        neg_atts = [self.idx2att[int(idx)]
                    for idx in (encoding == 0).nonzero(as_tuple=True)[0]
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
        image = sample.crop_image(image)
        if self.augmentations:
            # random grayscale when an instance is not
            # labeled with any color attributes
            if set(sample.positive_attributes).isdisjoint(
                    self.parent2attrib['color']):
                image = self.random_grayscale(image)
            image = self.augmentations(image)

        output = {"image": image,
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
                         attrib_parent_types="data/attribute_parent_types.json",
                         attrib_types="data/attribute_types.json",
                         img_dir="VG_100K",
                         split="test",
                         resize=224)
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=0,
        pin_memory=True,
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        for k in batch.keys():
            if type(batch[k]) == torch.Tensor:
                batch[k] = batch[k].to(device)
        if batch["object_name"][0] == "floor":
            print(dataset.decode_objs(batch["object_id"][0]))
            print(batch["image"].shape, batch["image"].device)
            # ToPILImage()(batch["image"][0]).show()
            continue
        pass

