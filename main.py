import os
from diffusers import StableDiffusion3Pipeline
from datasets import Dataset, DatasetDict
from PIL import Image
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from huggingface_hub import login
import pandas as pd
HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")

login("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")
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)

pipe.to("cuda")

# Assuming you have a simple directory structure for images and captions
images_path = "./images"
captions_path = "./captions"

# Load the images and captions into a Dataset
def load_dataset(images_path, captions_path):
    images = []
    captions = []
    for file in os.listdir(images_path):
        if file.endswith(".jpg") or file.endswith(".png"): # Adjust based on your image file types
            img_path = os.path.join(images_path, file)
            caption_path = os.path.join(captions_path, file.split('.')[0] + '.txt')
            if os.path.isfile(caption_path):
                images.append(img_path)
                with open(caption_path, 'r') as f:
                    captions.append(f.read())
    
    dataset = Dataset.from_pandas(pd.DataFrame({"image": images, "caption": captions}))
    return dataset

dataset = load_dataset(images_path, captions_path)

# Preprocess the dataset
def preprocess(examples):
    images = [Image.open(path) for path in examples["image"]]
    inputs = pipe.feature_extractor(images=images, captions=examples["caption"], return_tensors="pt")
    return inputs

dataset_dict = DatasetDict({"train": dataset})
dataset_dict = dataset_dict.map(preprocess, batched=True)


#Start finetuning

batch_size = 4  # Adjust based on your GPU's capacity
data_loader = DataLoader(dataset_dict["train"], batch_size=batch_size, shuffle=True)

# Set the model for training
pipe.to("cuda")
pipe.train()

# Define optimizer
optimizer = Adam(pipe.parameters(), lr=1e-5)

# Start the training loop
for epoch in range(3):
    for batch in data_loader:
        
        for k, v in batch.items():
            batch[k] = v.to("cuda")
        
        # Forward pass
        optimizer.zero_grad()
        outputs = pipe(**batch)
        loss = outputs.loss
        loss.backward()
        
        # Update parameters
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

pipe.save_pretrained("./fine_tuned_model")
