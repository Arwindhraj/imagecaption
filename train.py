from torch import optim
from torch.utils.data import Subset
from transformers import ViTImageProcessor, ViTModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from loss_plot import loss_plot
from model import *
from utils import *
from data_load import *

import os
import torch
import logging
import torch.nn as nn
import torch.utils.data as data
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
            outputs = model(attended_features, captions[:, :-1]) 
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # prevent exploding gradients, stabilizes training
            optimizer.step()

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

        print("Model Evaluating")
        logging.info("Model Evaluating")
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, captions in val_loader:
                images, captions = images.to(device), captions.to(device)
                outputs = model(images, captions[:, :-1])
                loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
    loss_plot(num_epochs, train_losses, val_losses)
    print("Training completed")
    logging.info("Training completed")

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
    
    batch_size = 64
    shuffle=True
    num_workers=8
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
    
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"], label_smoothing=0.1) # improve model calibration and generalization by preventing the model from becoming overconfident.

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    processor = ViTImageProcessor.from_pretrained('vit-base-patch16-224')
    model_vit = ViTModel.from_pretrained('vit-base-patch16-224').to(device)
    
    model_re = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model_re.eval()

    num_epochs = 30
    train_model(processor,model_vit,model_re, model, train_loader, val_loader, criterion, optimizer, num_epochs, device, scheduler)