import streamlit as st
from torchvision import datasets, transforms, models
from PIL import Image
import torch
import torch.nn as nn

class_name = ['Front Breakage', 'Front Crushed', 'Front Normal', 'Rear Breakage', 'Rear Crushed', 'Rear Normal']

class CarDamageClassResNet50(nn.Module):
    def __init__(self, num_class, drop_out):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Keep the original fc layer from ResNet50
        # Add a classifier layer to match the saved model
        self.model.classifier = nn.Sequential(
            nn.Dropout(drop_out),
            nn.Linear(self.model.fc.in_features, num_class)
        )

    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_model():
    model = CarDamageClassResNet50(num_class=6, drop_out=0.2)
    model.load_state_dict(torch.load("saved_model.pth", map_location='cpu'))
    model.eval()
    return model

def predict_from_image(image):
    """Predict damage from PIL Image object"""
    # Load model
    model = load_model()
    
    # Prepare transforms
    transform = transforms.Compose([
        transforms.Resize((280, 280)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert image and predict
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    
    return class_name[predicted.item()]

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((280, 280)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    model = load_model()

    with torch.no_grad():
        output = model(image_tensor)
    _, predicted = torch.max(output, 1)

    return class_name[predicted.item()]
