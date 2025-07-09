import streamlit as st
import io
import os

# Global variables
trained_model = None
class_name = ['Front Breakage', 'Front Crushed', 'Front Normal', 'Rare Breakage', 'Rare Crushed', 'Rare Normal']

class CarDamageClassResNet50:
    def __init__(self, num_class, drop_out):
        # Import torch modules directly inside the function
        import torch
        import torch.nn as nn
        from torchvision import models
        
        class _CarDamageClassResNet50(nn.Module):
            def __init__(self, num_class, drop_out):
                super().__init__()
                self.model = models.resnet50(weights='DEFAULT')

                for param in self.model.parameters():
                    param.requires_grad = False

                for param in self.model.layer4.parameters():
                    param.requires_grad = True

                self.model.classifier = nn.Sequential(
                    nn.Dropout(drop_out),
                    nn.Linear(self.model.fc.in_features, num_class)
                )

            def forward(self, x):
                return self.model(x)
        
        self.model = _CarDamageClassResNet50(num_class, drop_out)
    
    def __call__(self, x):
        return self.model(x)
    
    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)
    
    def eval(self):
        return self.model.eval()

@st.cache_resource
def load_model():
    """Load model with delayed PyTorch import"""
    global trained_model
    
    if trained_model is not None:
        return trained_model
    
    try:
        # Import torch directly inside the function
        import torch
        
        model = CarDamageClassResNet50(num_class=6, drop_out=0.2)
        
        model_path = "model\saved_model.pth"
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            trained_model = model
            st.success("✅ Model loaded successfully!")
            return model
        else:
            st.error("❌ Model file 'saved_model.pth' not found in the app directory.")
            st.info("Please upload your trained model file to deploy the app.")
            return None
            
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def predict_from_image(image):
    """Predict damage from PIL Image object"""
    try:
        # Import torch modules directly inside the function
        import torch
        from torchvision import transforms
        
        model = load_model()
        if model is None:
            return "Model not available"
        
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
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Prediction failed"

def predict(image_path):
    """Legacy function for compatibility"""
    try:
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        return predict_from_image(image)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return "Image loading failed"
