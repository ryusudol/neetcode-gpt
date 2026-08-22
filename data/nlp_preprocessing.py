import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        voca = set()
        for (p_sent, n_sent) in zip(positive, negative):
            for word in p_sent.split(): voca.add(word)
            for word in n_sent.split(): voca.add(word)
        voca = { item: idx + 1 for idx, item in enumerate(sorted(voca)) }

        encodings, max_len = [], 0
        for sent in positive:
            enc = []
            for word in sent.split(): enc.append(voca[word])
            encodings.append(enc.copy())
            max_len = max(max_len, len(enc))
        for sent in negative:
            enc = []
            for word in sent.split(): enc.append(voca[word])
            encodings.append(enc.copy())
            max_len = max(max_len, len(enc))
        
        for enc in encodings:
            if len(enc) < max_len:
                while len(enc) < max_len:
                    enc.append(0.0)
        
        # nn.utils.rnn.pad_sequence(torch.tensor(encodings))

        return encodings

        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)