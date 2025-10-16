"""
Inference class for MRI Orientation Classification.
Takes a NIfTI file and predicts the orientation of a single slice.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
from pathlib import Path
from torchvision import transforms
from PIL import Image

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_CHECKPOINT = "mri_orientation_finetuned.pth"
MRI_FOUNDATION_PATH = "C:/Users/user/Desktop/orientation code/mri_foundation-master"  # Path to cloned MRI Foundation repo
IMAGE_SIZE = (1024, 1024)
NUM_CLASSES = 3

LABEL_MAP = {0: 'Axial', 1: 'Coronal', 2: 'Sagittal'}

# ============================================================================
# MODEL CLASS (same as training)
# ============================================================================

class MRISAMClassifier(nn.Module):
    """Wrapper to convert SAM model to a 3-class classifier"""
    
    def __init__(self, base_model, num_classes=3, freeze_encoder=True):
        super().__init__()
        self.image_encoder = base_model.image_encoder
        self.num_classes = num_classes
        
        # Get encoder output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(
                next(base_model.image_encoder.parameters()).device
            )
            encoder_out = self.image_encoder(dummy_input)
            encoder_dim = encoder_out.shape[1]
        
        # Replace classification head
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(encoder_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Freeze encoder
        if freeze_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        # Pass through encoder
        features = self.image_encoder(x)
        # Pass through classification head
        logits = self.classification_head(features)
        return logits

# ============================================================================
# MAIN INFERENCE CLASS
# ============================================================================

class MRIOrientationClassifier:
    """
    Inference class for predicting MRI image orientation.
    Loads a checkpoint and predicts the orientation of a single slice.
    """
    
    def __init__(self, checkpoint_path, mri_foundation_path=None, device=None, image_size=IMAGE_SIZE):
        """
        Initialize the classifier.
        
        Args:
            checkpoint_path (str): Path to the fine-tuned checkpoint
            mri_foundation_path (str): Path to cloned MRI Foundation repo (required for SAM)
            device (str): Device to run inference on ('cuda' or 'cpu'). Defaults to CUDA if available.
            image_size (tuple): Input image size (height, width)
        """
        self.checkpoint_path = checkpoint_path
        self.image_size = image_size
        self.label_map = LABEL_MAP
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Add MRI Foundation to path
        if mri_foundation_path:
            import sys
            sys.path.insert(0, mri_foundation_path)
        
        # Load model
        self.model = self._load_model()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self):
        """Load the fine-tuned model from checkpoint"""
        print(f"Loading model from: {self.checkpoint_path}")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        try:
            from models.sam import sam_model_registry
            import cfg
            
            print("Loading SAM-based model...")
            
            # Load args using the cfg module
            args = cfg.parse_args()
            
            # Override key settings
            args.if_encoder_adapter = False
            args.if_mask_decoder_adapter = False
            args.num_cls = NUM_CLASSES
            args.image_size = self.image_size[0]
            
            # Ensure all required attributes exist
            if not hasattr(args, 'decoder_adapt_depth'):
                args.decoder_adapt_depth = 1
            if not hasattr(args, 'encoder_adapter_stride'):
                args.encoder_adapter_stride = 16
            if not hasattr(args, 'encoder_adapter_input_dim'):
                args.encoder_adapter_input_dim = 768
            if not hasattr(args, 'encoder_adapter_output_dim'):
                args.encoder_adapter_output_dim = 256
            
            # Create base SAM model WITHOUT loading checkpoint
            base_model = sam_model_registry["vit_b"](
                args,
                checkpoint=None,
                num_classes=NUM_CLASSES,
                image_size=self.image_size[0],
                pretrained_sam=False
            )
            base_model.to(self.device)
            
            # Wrap with classifier
            model = MRISAMClassifier(base_model, num_classes=NUM_CLASSES, freeze_encoder=False)
            
            # Load the checkpoint weights
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            print(" SAM model loaded successfully")
            return model
            
        except Exception as e:
            print(f" Error loading SAM model: {e}")
            print("Falling back to ResNet50...")
            from torchvision.models import resnet50
            
            model = resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint, strict=False)
            model.to(self.device)
            model.eval()
            print(" ResNet model loaded successfully")
            return model
    
    def predict_from_nifti(self, nifti_path, slice_index=None):
        """
        Predict orientation from a NIfTI file.
        
        Args:
            nifti_path (str): Path to the NIfTI file
            slice_index (int): Index of the slice to predict. If None, uses middle slice.
        
        Returns:
            dict: Contains 'orientation' (label name), 'label' (class index), 
                  and 'confidence' (softmax probability)
        """
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("nibabel is required. Install it with: pip install nibabel")
        
        print(f"Loading NIfTI file: {nifti_path}")
        
        if not os.path.exists(nifti_path):
            raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")
        
        # Load NIfTI file
        nifti = nib.load(nifti_path)
        data = nifti.get_fdata()
        
        print(f"NIfTI shape: {data.shape}")
        
        # Select slice
        if slice_index is None:
            # Use middle slice
            slice_index = data.shape[2] // 2
        
        if slice_index >= data.shape[2]:
            raise ValueError(f"Slice index {slice_index} out of range. Max: {data.shape[2]-1}")
        
        # Extract slice
        slice_data = data[:, :, slice_index]
        
        print(f"Using slice {slice_index}")
        
        # Normalize to [0, 1]
        slice_min = slice_data.min()
        slice_max = slice_data.max()
        if slice_max > slice_min:
            slice_data = (slice_data - slice_min) / (slice_max - slice_min)
        
        # Convert to image for prediction
        prediction = self.predict_from_array(slice_data)
        
        return prediction
    
    def predict_from_array(self, image_array):
        """
        Predict orientation from a numpy array (2D image).
        
        Args:
            image_array (np.ndarray): 2D numpy array representing an image
        
        Returns:
            dict: Contains 'orientation' (label name), 'label' (class index), 
                  and 'confidence' (softmax probability)
        """
        # Convert to PIL Image
        if image_array.dtype == np.uint8:
            img = Image.fromarray(image_array, mode='L')
        else:
            # Convert to 0-255 range
            img_uint8 = (image_array * 255).astype(np.uint8)
            img = Image.fromarray(img_uint8, mode='L')
        
        # Convert grayscale to RGB
        img_rgb = img.convert('RGB')
        
        # Resize if necessary
        if img_rgb.size != self.image_size:
            img_rgb = img_rgb.resize(self.image_size, Image.BILINEAR)
        
        # Apply transforms
        img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_label = torch.max(probabilities, 1)
        
        predicted_label = predicted_label.item()
        confidence = confidence.item()
        orientation = self.label_map[predicted_label]
        
        result = {
            'orientation': orientation,
            'label': predicted_label,
            'confidence': confidence,
            'all_probabilities': {
                self.label_map[i]: probabilities[0, i].item()
                for i in range(NUM_CLASSES)
            }
        }
        
        return result
    
    def predict_from_image(self, image_path):
        """
        Predict orientation from an image file.
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            dict: Contains 'orientation' (label name), 'label' (class index), 
                  and 'confidence' (softmax probability)
        """
        print(f"Loading image: {image_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        img = Image.open(image_path).convert('RGB')
        
        # Resize if necessary
        if img.size != self.image_size:
            img = img.resize(self.image_size, Image.BILINEAR)
        
        # Apply transforms
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_label = torch.max(probabilities, 1)
        
        predicted_label = predicted_label.item()
        confidence = confidence.item()
        orientation = self.label_map[predicted_label]
        
        result = {
            'orientation': orientation,
            'label': predicted_label,
            'confidence': confidence,
            'all_probabilities': {
                self.label_map[i]: probabilities[0, i].item()
                for i in range(NUM_CLASSES)
            }
        }
        
        return result


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize classifier
    print(torch.cuda.is_available())
    classifier = MRIOrientationClassifier(
        checkpoint_path=MODEL_CHECKPOINT,
        mri_foundation_path=MRI_FOUNDATION_PATH,
        device="cpu"  # or "cpu"
    )

    axial_count = 0
    coronal_count = 0
    sagittal_count = 0

    # # Example 1: Predict from NIfTI file
    # nifti_file = r"C:\Users\user\Desktop\Dataset002_Heart\imagesTr\la_024.nii.gz"
    # img = nib.load(nifti_file)
    # for i in range(0 , img.shape[2], 40):
    #     print(f"\nPredicting for NIfTI slice {i}")
    #     result = classifier.predict_from_nifti(nifti_file, slice_index=i)
    #     if result['orientation'] == 'Axial':
    #         axial_count += 1
    #     if result['orientation'] == 'Coronal':
    #         coronal_count += 1
    #     if result['orientation'] == 'Sagittal':
    #         sagittal_count += 1
    #     print(f"\nPrediction Results:")
    #     print(f"  Orientation: {result['orientation']}")
    #     print(f"  Confidence: {result['confidence']:.4f}")
    #     print(f"  All probabilities: {result['all_probabilities']}")
    
    # Example 2: Predict from numpy array
    # dummy_array = np.random.rand(512, 512)
    # result = classifier.predict_from_array(dummy_array)
    
    # Example 3: Predict from image file
    from random import choice
    image_files = list(Path('drive-download-20251016T102446Z-1-001').glob("*.png"))
    for i in range(10):
        image_file = choice(image_files)
        print(f"\nPredicting for image: {image_file}")
        result = classifier.predict_from_image(image_file)
        if result['orientation'] == 'Axial':
            axial_count += 1
        if result['orientation'] == 'Coronal':
            coronal_count += 1
        if result['orientation'] == 'Sagittal':
            sagittal_count += 1 
        print(f"\nPrediction Results:")
        print(f"  Orientation: {result['orientation']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  All probabilities: {result['all_probabilities']}")

    print(f"\nFinal Counts:\nAxial: {axial_count}\nCoronal: {coronal_count}\nSagittal: {sagittal_count}")

    counts = {
        'Axial': axial_count,
        'Coronal': coronal_count,
        'Sagittal': sagittal_count
    }

    predicted_orientation = max(counts, key=counts.get)
    print(f'Final Prediction : {predicted_orientation}')