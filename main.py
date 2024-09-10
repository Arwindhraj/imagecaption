from PIL import Image
from torch import optim
from collections import Counter
from prettytable import PrettyTable
from torch.utils.data import Subset
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import functional as F
from sklearn.model_selection import train_test_split
from transformers import ViTImageProcessor, ViTModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torchvision.models.detection import fasterrcnn_resnet50_fpn

import os
import math
import nltk
import torch
import logging
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def setup_logging():
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

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
    
def Vit(image,processor,model_vit):

    inputs = processor(images=image, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model_vit(**inputs)

    features = outputs.last_hidden_state[:, 0, :]

    return features

def res(image,model_re):

    image_tensor = image  

    with torch.no_grad():
        outputs = model_re(image_tensor)

    boxes = outputs[0]['boxes']  
    labels = outputs[0]['labels']  
    scores = outputs[0]['scores']  

    return outputs, boxes, labels, scores

@torch.no_grad()
def combine_features(image,processor,model_vit,model_re,):

    vit_features = Vit(image,processor,model_vit)
    _, boxes, labels, scores = res(image,model_re)
    
    combined_features = {
        'vit': vit_features,
        'boxes': boxes,
        'labels': labels,
        'scores': scores
    }
    combined_features = combined_features
    return combined_features

class LocalFeatureEncoder(nn.Module):
    def __init__(self, input_dim=5, d_model=768, nhead=8, num_encoder_layers=6):
        super(LocalFeatureEncoder, self).__init__()
        self.d_model = d_model
        self.rcnn_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 10, d_model)) 
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, rcnn_features):
        rcnn_features = rcnn_features.to("cuda")
        x = self.rcnn_projection(rcnn_features)
        x = x + self.pos_encoder
        
        encoded_features = self.transformer_encoder(x)
        encoded_features = self.norm(encoded_features)
        return encoded_features

class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attn_output, _ = self.multihead_attn(query, key, value)

        return self.dropout(self.norm(query + attn_output))

def apply_cross_attention(vit_features, encoded_local_features):
    if vit_features.dim() == 2:
        vit_features = vit_features.unsqueeze(0)
    if encoded_local_features.dim() == 2:
        encoded_local_features = encoded_local_features.unsqueeze(0)

    vit_features = vit_features.transpose(0, 1).to("cuda")
    encoded_local_features = encoded_local_features.transpose(0, 1).to("cuda")
    
    attended_features = cross_attention(vit_features, encoded_local_features, encoded_local_features)
    
    attended_features = attended_features.transpose(0, 1)
    attended_features = attended_features.to("cuda")
    
    return attended_features

def prepare_for_cross_attention(combined_features):
    vit_features = combined_features['vit'].clone().detach().unsqueeze(0).to("cuda")

    
    boxes = combined_features['boxes']
    labels = combined_features['labels']
    scores = combined_features['scores']
    
    confidence_threshold = 0.5
    high_confidence_indices = scores > confidence_threshold
    
    filtered_boxes = boxes[high_confidence_indices]
    filtered_labels = labels[high_confidence_indices]
    
    rcnn_features = torch.cat([filtered_boxes, filtered_labels.unsqueeze(1).float()], dim=1)
    rcnn_features = rcnn_features.to("cuda")
    max_objects = 10
    if rcnn_features.size(0) < max_objects:
        padding = torch.zeros(max_objects - rcnn_features.size(0), rcnn_features.size(1))
        padding = padding.to("cuda")
        rcnn_features = torch.cat([rcnn_features, padding], dim=0)
    else:
        rcnn_features = rcnn_features[:max_objects]
    
    rcnn_features = rcnn_features.unsqueeze(0)
    rcnn_features = rcnn_features.to("cuda")
    
    encoded_local_features = local_feature_encoder(rcnn_features.to("cuda"))
    encoded_local_features = encoded_local_features.to("cuda")
    
    attended_features = apply_cross_attention(vit_features, encoded_local_features)
    
    return attended_features

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=8, num_decoder_layers=6, dropout=0.1):    
        super(Decoder, self).__init__()        
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attended_features_projection = nn.Linear(768, d_model) # To handle potential shape mismatch

    # def forward(self, images, captions):
    #     images = images.to("cuda")
    #     print(f"Images shape: {images.shape}")
    #     print(f"Captions shape: {captions.shape}")
        
    #     attended_features = self.attended_features_projection(images)
    #     print(f"Attended features shape after projection: {attended_features.shape}")
        
    #     embedded_captions = self.embedding(captions)
    #     print(f"Embedded captions shape: {embedded_captions.shape}")
        
    #     embedded_captions = self.norm(embedded_captions)
    #     embedded_captions = self.dropout(embedded_captions)
        
    #     # Adjust dimensions for the decoder
    #     attended_features = attended_features.transpose(0, 1)
    #     embedded_captions = embedded_captions.transpose(0, 1)
        
    #     print(f"Attended features shape before decoder: {attended_features.shape}")
    #     print(f"Embedded captions shape before decoder: {embedded_captions.shape}")
        
    #     decoded_features = self.decoder(embedded_captions, attended_features)
    #     print(f"Decoded features shape: {decoded_features.shape}")
        
    #     outputs = self.fc_out(decoded_features.transpose(0, 1))
    #     print(f"Outputs shape: {outputs.shape}")
        
    #     return outputs
    def forward(self, images, captions):
        print(f"Images shape: {images.shape}")
        print(f"Captions shape: {captions.shape}")
        
        # Project and reshape attended features
        attended_features = self.attended_features_projection(images)
        attended_features = attended_features.squeeze(0).unsqueeze(1).repeat(1, captions.size(1), 1)
        print(f"Attended features shape after projection and reshape: {attended_features.shape}")
        
        embedded_captions = self.embedding(captions)
        print(f"Embedded captions shape: {embedded_captions.shape}")
        
        embedded_captions = self.norm(embedded_captions)
        embedded_captions = self.dropout(embedded_captions)
        
        # Adjust dimensions for the decoder
        attended_features = attended_features.transpose(0, 1)
        embedded_captions = embedded_captions.transpose(0, 1)
        
        print(f"Attended features shape before decoder: {attended_features.shape}")
        print(f"Embedded captions shape before decoder: {embedded_captions.shape}")
        
        decoded_features = self.decoder(embedded_captions, attended_features)
        print(f"Decoded features shape: {decoded_features.shape}")
        
        outputs = self.fc_out(decoded_features.transpose(0, 1))
        print(f"Outputs shape: {outputs.shape}")
        
        return outputs
        

def train_model(processor, model_vit, model_re, model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler):
    model.to("cuda")
    checkpoint_dir = 'checkpoint'
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (images, captions) in enumerate(train_loader):
            images, captions = images.to("cuda"), captions.to("cuda")

            combined_features = combine_features(images,processor,model_vit,model_re)
            attended_features = prepare_for_cross_attention(combined_features)
            
            optimizer.zero_grad()
            print("Image --> Model for ",batch_idx)
            outputs = model(attended_features, captions[:, :-1]) 
            print("Calculating Loss ▲ for ",batch_idx)
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # prevent exploding gradients, stabilizes training
            optimizer.step()
            print("Loss calculated ✔ for ",batch_idx)
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
                logging.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_train_loss:.4f}")
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_train_loss:.4f}")

        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_train_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        logging.info(f"Checkpoint saved to {checkpoint_path}")

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, captions in val_loader:
                images, captions = images.to("cuda"), captions.to("cuda")
                outputs = model(images, captions[:, :-1])
                loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.savefig('training_loss_plot.png')
    plt.close()
    print("Training Loss plot saved as 'training_loss_plot.png'")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Evaluate Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Evaluate Loss over Epochs')
    plt.legend()
    plt.savefig('evaluate_loss_plot.png')
    plt.close()
    print("Evaluate Loss plot saved as 'evaluate_loss_plot.png'")
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.savefig('train_valid_loss.png')
    plt.close()
    print("Loss plot saved as 'train_valid_loss.png'")
    
    print("Training completed")
    logging.info("Training completed")
    

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

local_feature_encoder = LocalFeatureEncoder(input_dim=5, d_model=768).to("cuda")
cross_attention = CrossAttention(d_model=768, nhead=8).to("cuda")

if __name__ == "__main__":

    setup_logging()
    logging.info("Training Started")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    root_folder="flicker30k/Images"
    annotation_file="flicker30k/captions.txt"
    batch_size = 32
    shuffle=True
    num_workers=4
    pin_memory=True

    dataset = Flickr30kDataset(root_folder, annotation_file, transform=transform)

    train_indices, val_indices = dataset.split_dataset()

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    pad_idx = dataset.vocab.stoi["<PAD>"]
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )
    val_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )

    vocab_size = len(dataset.vocab)
    model = Decoder(vocab_size)
    
    count_parameters(model)

    # total_parameter = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Decoder model parameter count",total_parameter)
    
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"], label_smoothing=0.1)# improve model calibration and generalization by preventing the model from becoming overconfident.

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    processor = ViTImageProcessor.from_pretrained('vit-base-patch16-224')
    model_vit = ViTModel.from_pretrained('vit-base-patch16-224').to(device)
    
    model_re = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model_re.eval()

    num_epochs = 30
    train_model(processor,model_vit,model_re, model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler)