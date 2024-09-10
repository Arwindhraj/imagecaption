from model import *
from prettytable import PrettyTable

import torch

local_feature_encoder = LocalFeatureEncoder(input_dim=5, d_model=768)
cross_attention = CrossAttention(d_model=768, nhead=8)

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