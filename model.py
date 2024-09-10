from torch.nn import TransformerDecoder, TransformerDecoderLayer

import torch
import torch.nn as nn

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

    def forward(self, images, captions):
        
        attended_features = self.attended_features_projection(images)
        attended_features = attended_features.squeeze(0).unsqueeze(1).repeat(1, captions.size(1), 1) # Project and reshape attended features
        
        embedded_captions = self.embedding(captions)
        embedded_captions = self.norm(embedded_captions)
        embedded_captions = self.dropout(embedded_captions)
    
        attended_features = attended_features.transpose(0, 1)
        embedded_captions = embedded_captions.transpose(0, 1)
        
        decoded_features = self.decoder(embedded_captions, attended_features)
        
        outputs = self.fc_out(decoded_features.transpose(0, 1))
        
        return outputs