from torchvision.datasets import Flickr30k
from torchvision import transforms
from torch.utils.data import DataLoader

if __name__ == "__main__":
    root_folder = "archive/Images"
    annotation_file = "archive/captions.txt"
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    ])

    dataset = Flickr30k(
        root=root_folder,
        ann_file=annotation_file,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True) 
    print(f"Total number of samples in the dataset: {len(dataset)}")
    
    for batch in dataloader:
        images, captions = batch
        print(f"Batch size: {len(images)}")
        print(f"Images shape: {images.shape}")
        print(f"Number of caption pairs: {len(captions)}")
        
        # Initialize lists to store captions for each image
        image1_captions = []
        image2_captions = []
        
        # Organize captions
        for caption_pair in captions:
            image1_captions.append(caption_pair[0])
            image2_captions.append(caption_pair[1])
        
        # Print captions for each image
        for i, image_captions in enumerate([image1_captions, image2_captions], 1):
            print(f"\nImage {i}:")
            print(f"Number of captions: {len(image_captions)}")
            for j, caption in enumerate(image_captions, 1):
                print(f"  Caption {j}: {caption}")
        
        break
