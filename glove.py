import pickle
import typing as t
import re
import os

import torch
import torch.nn as nn

class GloVeEmbedding(nn.Module):
    def __init__(self, glove_file_path: os.path, load_cached: bool = True) -> None:
        super(GloVeEmbedding, self).__init__()
        self.glove_file_path = glove_file_path
        self.load_cached = load_cached

        self.words = []
        
        if not os.path.exists("data/glove.pkl"):
            self.load_cached = False

        if self.load_cached:
            with open("data/glove.pkl", "rb") as f:
                data = pickle.load(f)
            self.words = data["words"]
            self.vectors = data["vectors"]
        else:
            with open(self.glove_file_path, "r") as f:
                data = f.readlines()
            vectors = []
            for line in data:
                line = line.split()
                word, vector = line[0], torch.tensor([float(val) for val in line[1:]])
                self.words.append(word)
                vectors.append(vector)
            self.vectors = nn.Parameter(torch.stack(vectors, dim=0))
            with open("data/glove.pkl", "wb") as f:
                pickle.dump({"words": self.words, "vectors": self.vectors}, f)

    def embed_object(self, obj: str) -> torch.Tensor:
        obj_embedding = torch.zeros_like(self.vectors[0])
        for word in re.split(' |-', obj):
            try:
                obj_embedding += self.vectors[self.words.index(word.lower())]
            except ValueError:
                # random in the range [-0.04, 0.04) if 
                # object name is not in GloVe embedding
                obj_embedding += (0.04 - -0.04)*torch.rand_like(self.vectors[0]) + -0.04
        obj_embedding /= len(re.split(' |-', obj))

        return obj_embedding
            
    def embed_objects(self, objs: t.List) -> torch.Tensor:
        return torch.stack([self.embed_object(obj) for obj in objs])

    def forward(self, x: t.Union[t.List[str], str]) -> torch.Tensor:
        """Provides the GloVe embedding for a string or a list of strings"""
        if isinstance(x, str):
            return torch.stack([self.embed_object(x)])
        else:
            return self.embed_objects(x)

if __name__ == "__main__":
    glove = GloVeEmbedding("data/glove.6B.100d.txt")
    print(glove("floor"))
    print(glove(["floor", "ceiling", "brown haired"]))
