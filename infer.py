from PIL import Image
from rouge import Rouge
from model import Decoder
from data_load import Flickr30kDataset
from nltk.translate.bleu_score import corpus_bleu
from transformers import ViTImageProcessor, ViTModel
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from utils import combine_features, prepare_for_cross_attention
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import os
import torch
import logging
import argparse
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def setup_logging():
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'inference.log')),
            logging.StreamHandler()
        ]
    )

def load_model(model_path, vocab_size):
    model = Decoder(vocab_size)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def generate_caption(model, image, processor, model_vit, model_re, dataset, max_length=50):
    combined_features = combine_features(image, processor, model_vit, model_re)
    attended_features = prepare_for_cross_attention(combined_features)

    caption = [dataset.vocab.stoi["< SOS >"]]
    
    for _ in range(max_length):
        caption_tensor = torch.LongTensor(caption).unsqueeze(0).to(device)
        output = model(attended_features, caption_tensor)
        predicted = output.argmax(2)[:, -1].item()
        caption.append(predicted)
        
        if predicted == dataset.vocab.stoi["<EOS>"]:
            break
    
    return [dataset.vocab.itos[idx] for idx in caption]

def display_result(image_path, caption):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(Image.open(image_path))
    plt.axis('off')
    plt.title('Input Image')
    
    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.5, ' '.join(caption[1:-1]), fontsize=12, ha='center', va='center', wrap=True)
    plt.axis('off')
    plt.title('Generated Caption')
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, val_loader, processor, model_vit, model_re, dataset):
    model.eval()
    references = []
    hypotheses = []
    all_true_labels = []
    all_pred_labels = []
    
    with torch.no_grad():
        for images, captions in val_loader:
            images = images.to(device)
            for i in range(images.size(0)):
                image = images[i].unsqueeze(0)
                reference = [dataset.vocab.itos[idx.item()] for idx in captions[i] if idx.item() not in [dataset.vocab.stoi["<PAD>"], dataset.vocab.stoi["< SOS >"], dataset.vocab.stoi["<EOS>"]]]
                hypothesis = generate_caption(model, image, processor, model_vit, model_re, dataset)[1:-1]  # Remove SOS and EOS tokens
                
                references.append([reference])
                hypotheses.append(hypothesis)
                
                all_true_labels.extend(reference)
                all_pred_labels.extend(hypothesis)
    
    bleu_score = corpus_bleu(references, hypotheses)
    
    rouge = Rouge()
    rouge_scores = rouge.get_scores([' '.join(h) for h in hypotheses], [' '.join(r[0]) for r in references], avg=True)
    
    accuracy = accuracy_score(all_true_labels, all_pred_labels)
    precision = precision_score(all_true_labels, all_pred_labels, average='weighted', zero_division=1)
    recall = recall_score(all_true_labels, all_pred_labels, average='weighted', zero_division=1)
    f1 = f1_score(all_true_labels, all_pred_labels, average='weighted', zero_division=1)
    
    return bleu_score, rouge_scores, accuracy, precision, recall, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Caption Generation")
    parser.add_argument("--mode", choices=["new", "validation"], default="new", help="Inference mode: 'new' for new images, 'validation' for validation dataset")
    parser.add_argument("--image_path","--imgpath", help="Path to the new image for captioning (required for 'new' mode)")
    parser.add_argument("--num_val_samples","--n", type=int, default=5, help="Number of validation samples to caption (for 'validation' mode)")
    args = parser.parse_args()

    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    root_folder = "images"
    annotation_file = "captions.txt"
    dataset = Flickr30kDataset(root_folder, annotation_file)
    
    model_path = "model.pth"  
    model = load_model(model_path, len(dataset.vocab)).to(device)
    
    processor = ViTImageProcessor.from_pretrained('vit-base-patch16-224')
    model_vit = ViTModel.from_pretrained('vit-base-patch16-224').to(device)
    model_re = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model_re.eval()

    if args.mode == "new":
        if not args.image_path:
            raise ValueError("Image path is required for 'new' mode")
        
        image = preprocess_image(args.image_path).to(device)
        caption = generate_caption(model, image, processor, model_vit, model_re, dataset)
        display_result(args.image_path, caption)
        print("Generated Caption:", ' '.join(caption[1:-1]))

    elif args.mode == "validation":
        val_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.Subset(dataset, dataset.split_dataset()[1]),
            batch_size=1,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        bleu_score, rouge_scores, accuracy, precision, recall, f1 = evaluate_model(model, val_loader, processor, model_vit, model_re, dataset)
        
        print(f"BLEU Score: {bleu_score}")
        print(f"ROUGE Scores: {rouge_scores}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        
        logging.info(f"BLEU Score: {bleu_score}")
        logging.info(f"ROUGE Scores: {rouge_scores}")
        logging.info(f"Accuracy: {accuracy}")
        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F1 Score: {f1}")

        for i in range(args.num_val_samples):
            images, captions = next(iter(val_loader))
            image = images[0].to(device)
            image_name = os.path.basename(dataset.df['image'][dataset.split_dataset()[1][i]])
            true_caption = ' '.join([dataset.vocab.itos[idx.item()] for idx in captions[0] if idx.item() not in [dataset.vocab.stoi["<PAD>"], dataset.vocab.stoi["< SOS >"], dataset.vocab.stoi["<EOS>"]]])
            generated_caption = generate_caption(model, image, processor, model_vit, model_re, dataset)
            
            print(f"\nImage: {image_name}")
            print("True Caption:", true_caption)
            print("Generated Caption:", ' '.join(generated_caption[1:-1]))
            logging.info(f"Image: {image_name}")
            logging.info(f"True Caption: {true_caption}")
            logging.info(f"Generated Caption: {' '.join(generated_caption[1:-1])}")
            display_result(os.path.join(root_folder, image_name), generated_caption)

    else:
        raise ValueError("Invalid mode. Choose 'new' or 'validation'.")
