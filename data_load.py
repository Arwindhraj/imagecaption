from PIL import Image
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

import os
import nltk
import torch
import torch.utils.data as data

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        
    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer_eng(text):
        return nltk.word_tokenize(text.lower())
    
    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1
                
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
                    
    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokenized_text
        ]
    
class Flickr30kDataset(data.Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = self.load_captions(captions_file)
        self.transform = transform

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.df['caption'])
        
    def load_captions(self, captions_file):
        data = {'image': [], 'caption': []}
        with open(captions_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        for line in lines:
            image, caption = line.strip().split(',', 1)
            data['image'].append(image)
            data['caption'].append(caption)
        
        return data
    
    def __len__(self):
        return len(self.df['caption'])
    
    def __getitem__(self, index):
        caption = self.df['caption'][index]
        img_id = self.df['image'][index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
            
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])
        
        return img, torch.tensor(numericalized_caption)
    
    def split_dataset(self, val_split=0.2, random_state=42):
        indices = list(range(len(self)))
        val_size = int(len(self) * val_split)
        train_indices, val_indices = train_test_split(
            indices, test_size=val_size, random_state=random_state
        )
        return train_indices, val_indices

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
        
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        captions = [item[1] for item in batch]
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        
        return imgs, captions