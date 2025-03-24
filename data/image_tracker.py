import numpy as np
import torch
import cv2
import os
import requests
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from PIL import Image
from torchvision import models, transforms
from torchvision.models import EfficientNet_V2_L_Weights
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from stabilizer import Stabilizer

class ImageTracker:
    def __init__(
        self, 
        target_size: Tuple[int, int],
        stabilizer: Stabilizer,
        sam2_checkpoint: str,
        sam2_config: str
    ):
        self.target_size = target_size
        self.stabilizer = stabilizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # SAM2 configuration
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        
        # Initialize models
        self.sam = self._setup_sam2_model()
        self.classifier = self._setup_classifier()
        
        self.transform = transforms.Compose([
            transforms.Resize(480, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _setup_sam2_model(self) -> Any:
        if not os.path.exists(self.sam2_checkpoint):
            print(f"Downloading {self.sam2_checkpoint}...")
            url = f"https://dl.fbaipublicfiles.com/segment_anything_2/{self.sam2_checkpoint}"
            with open(self.sam2_checkpoint, "wb") as f:
                f.write(requests.get(url).content)
        
        if not os.path.exists(self.sam2_config):
            print(f"Downloading {self.sam2_config}...")
            url = f"https://raw.githubusercontent.com/facebookresearch/segment-anything-2/main/configs/sam2.1/{self.sam2_config}"
            with open(self.sam2_config, "wb") as f:
                f.write(requests.get(url).content)

        sam2 = build_sam2(f"/{os.path.abspath(self.sam2_config)}", self.sam2_checkpoint, device=self.device, apply_postprocessing=False)
        return sam2

    def _setup_classifier(self) -> torch.nn.Module:
        model = models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
        model.eval()
        model.to(self.device)
        return model

    def _load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _generate_masks(self, image: np.ndarray) -> List[Dict]:
        # Use SAM2AutomaticMaskGenerator for generating masks
        mask_generator = SAM2AutomaticMaskGenerator(self.sam)
        masks = mask_generator.generate(image)
        return sorted(masks, key=lambda x: x['area'], reverse=True)

    def _filter_masks(self, image: np.ndarray, masks: List[Dict]) -> List[Dict]:
        sock_masks = []
        image_height, image_width = image.shape[:2]
        min_area = 0.005 * image_height * image_width
        max_area = 0.1 * image_height * image_width
        
        for mask in masks:
            m = mask['segmentation']
            area = mask['area']
            
            if min_area <= area <= max_area:
                mask_binary = m.astype(np.uint8)
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    hull = cv2.convexHull(largest_contour)
                    hull_area = cv2.contourArea(hull)
                    
                    if hull_area > 0:
                        sock_masks.append(mask)
        
        return sock_masks
    
    def _classify_socks(self, image: np.ndarray, masks: List[Dict]) -> List[Dict]:
        sock_class_idx = 806
        confidence_threshold = 0.01
        confirmed_socks = []
        
        for mask in masks:
            # Extract the masked region
            mask_binary = mask['segmentation'].astype(np.uint8)
            y_indices, x_indices = np.where(mask_binary > 0)
            
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
                
            # Extract and pad the bounding box
            padding = 10
            min_y, max_y = np.min(y_indices), np.max(y_indices)
            min_x, max_x = np.min(x_indices), np.max(x_indices)
            min_y, min_x = max(0, min_y - padding), max(0, min_x - padding)
            max_y, max_x = min(image.shape[0], max_y + padding), min(image.shape[1], max_x + padding)
            roi = image[min_y:max_y, min_x:max_x].copy()
            roi_pil = Image.fromarray(roi)
            input_tensor = self.transform(roi_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.classifier(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                
            # Check if it's classified as a sock with enough confidence
            if probabilities[sock_class_idx] >= confidence_threshold:
                mask['sock_confidence'] = float(probabilities[sock_class_idx])
                confirmed_socks.append(mask)
        
        return confirmed_socks

    def visualize_masks(self, image: np.ndarray, masks: List[Dict], output_path: Optional[str] = None) -> np.ndarray:
        vis_image = image.copy()
        colors = plt.cm.tab20(np.linspace(0, 1, len(masks)))
        
        for i, mask in enumerate(masks):
            m = mask['segmentation']
            color = (colors[i % len(colors)][:3] * 255).astype(np.uint8)
            color_tuple = (int(color[0]), int(color[1]), int(color[2]))
            
            binary_mask = m.astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Apply semi-transparent mask
            colored_mask = np.zeros_like(vis_image)
            colored_mask[m] = color_tuple
            
            alpha = 0.2
            mask_area = binary_mask > 0
            vis_image[mask_area] = cv2.addWeighted(vis_image[mask_area], 1 - alpha, colored_mask[mask_area], alpha, 0)
            
            cv2.drawContours(vis_image, contours, -1, color_tuple, 2)
            
            if contours:
                M = cv2.moments(max(contours, key=cv2.contourArea))
                if M["m00"] != 0:
                    cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    conf_str = f"{mask.get('sock_confidence', 0):.2f}" if 'sock_confidence' in mask else ""
                    text_size, _ = cv2.getTextSize(conf_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    
                    cv2.rectangle(vis_image, 
                        (cX - text_size[0]//2 - 5, cY - text_size[1]//2 - 5),
                        (cX + text_size[0]//2 + 5, cY + text_size[1]//2 + 5),
                        color_tuple, -1
                    )
                    
                    cv2.putText(vis_image, conf_str, 
                        (cX - text_size[0]//2, cY + text_size[1]//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                    )
        
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        return vis_image

    def process_image(
        self, 
        image_path: str, 
        warped_dir: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        # Load and resize the original image
        original_image = self._load_image(image_path)
        image = cv2.resize(original_image, self.target_size, interpolation=cv2.INTER_AREA)
        
        # Apply perspective correction
        image, perspective_corrected, transform_matrix, mm_per_pixel = self.stabilizer.stabilize_frame(image)
        
        # Generate and filter masks using SAM2
        masks = self._generate_masks(image)
        sock_masks = self._filter_masks(image, masks)
        sock_masks = self._classify_socks(image, sock_masks)
        
        # Create visualization only if needed
        vis_image = None
        if 'VIZ' in os.environ:
            vis_image = self.visualize_masks(image, sock_masks)
        
        # Save outputs if required
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save perspective corrected image if applicable
        if perspective_corrected and warped_dir is not None:
            os.makedirs(warped_dir, exist_ok=True)
            warped_output = os.path.join(warped_dir, f"{base_name}.jpg")
            cv2.imwrite(warped_output, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        

    def process_directory(
        self, 
        input_dir: str, 
        annotation_dir: str,
        warped_dir: str
    ) -> None:
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return
        
        # Create output directories
        os.makedirs(annotation_dir, exist_ok=True)
        os.makedirs(warped_dir, exist_ok=True)
        
        for filename in tqdm(image_files, desc="Processing sock images"):
            image_path = os.path.join(input_dir, filename)
            self.process_image(
                image_path, 
                annotation_dir=annotation_dir,
                warped_dir=warped_dir
            )
        
        print(f"Processed {len(image_files)} images. Results saved.")

    def visualize_batch(
        self, 
        image_files: List[str], 
        input_dir: str
    ) -> None:
        num_images = min(3, len(image_files))
        if num_images == 0:
            print("No images found!")
            return
        
        _, axes = plt.subplots(num_images, 3, figsize=(15, 10))
        
        for i in range(num_images):
            image_path = os.path.join(input_dir, image_files[i])
            
            # Set VIZ env var temporarily for this call to generate visualization
            os.environ['VIZ'] = '1'
            
            processed_image, vis_image, sock_masks = self.process_image(image_path)
            
            # Reset VIZ env var if it wasn't set
            if os.environ.get('VIZ') != '1':
                os.environ.pop('VIZ', None)
            
            # Original image (for comparison)
            original_image = self._load_image(image_path)
            original_image = cv2.resize(original_image, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Display results
            axes[i, 0].imshow(original_image)
            axes[i, 0].set_title(f"Original - {os.path.basename(image_path)}", fontsize=12)
            axes[i, 0].axis('off')
            
            # Perspective corrected
            axes[i, 1].imshow(processed_image)
            axes[i, 1].set_title("Top-Down View (Scale Preserved)", fontsize=12)
            axes[i, 1].axis('off')
            
            # Sock detection
            axes[i, 2].imshow(vis_image)
            axes[i, 2].set_title(f"Detected - {len(sock_masks)} socks", fontsize=12)
            axes[i, 2].axis('off')

        plt.tight_layout() 
        plt.savefig("sock_detection_preview.jpg", dpi=150, bbox_inches='tight')
        print("Preview saved as sock_detection_preview.jpg")
        plt.show()


if __name__ == "__main__":
    input_dir = "sock_images"
    annotation_dir = "sock_annotation"
    warped_dir = "sock_warped"
    aruco_dict_type = cv2.aruco.DICT_6X6_250
    aruco_marker_id = 5
    output_warped_size = (800, 600)
    marker_size_mm = 80.0
    sam2_checkpoint = "sam2.1_hiera_large.pt"
    sam2_config = "sam2.1_hiera_l.yaml"
    
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"Created directory {input_dir} - please add your sock images there and run again")
        exit(0)
    
    # Create stabilizer
    stabilizer = Stabilizer(
        aruco_dict_type=aruco_dict_type,
        aruco_marker_id=aruco_marker_id,
        output_warped_size=output_warped_size,
        marker_size_mm=marker_size_mm
    )
    
    # Create image tracker
    image_tracker = ImageTracker(
        target_size=output_warped_size,
        stabilizer=stabilizer,
        sam2_checkpoint=sam2_checkpoint,
        sam2_config=sam2_config
    )
    
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        exit(0)
    
    # Always visualize when running this file directly
    image_tracker.visualize_batch(image_files, input_dir)