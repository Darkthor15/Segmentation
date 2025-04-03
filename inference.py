import os
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
import csv

with open('inference_config.json', 'r') as f:
    config = json.load(f)

priority_to_color = config["color_mapping"]
priority_to_color = {int(k): tuple(v) for k, v in priority_to_color.items()}
blend_alpha = config["blend_alpha"]

inference_dir = 'inference'
heatmap_dir = os.path.join(inference_dir, 'heatmaps')
segmentation_dir = os.path.join(inference_dir, 'segmentations')
overlay_dir = os.path.join(inference_dir, 'overlays')
actual_dir = os.path.join(inference_dir, 'actual')
os.makedirs(heatmap_dir, exist_ok=True)
os.makedirs(segmentation_dir, exist_ok=True)
os.makedirs(overlay_dir, exist_ok=True)
os.makedirs(actual_dir, exist_ok=True)

class InferenceDataset(Dataset):
    def __init__(self, root_dir, image_processor):
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.img_dir = os.path.join(self.root_dir, "images", "validation")
        self.ann_dir = os.path.join(self.root_dir, "annotations", "validation")
        self.images = sorted(os.listdir(self.img_dir))
        self.annotations = sorted(os.listdir(self.ann_dir))
        print(self.img_dir, self.images)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        annotation = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))
        encoded_inputs = self.image_processor(image, return_tensors="pt")
        encoded_inputs['image_name'] = self.images[idx]
        encoded_inputs['annotation'] = torch.tensor(np.array(annotation))
        encoded_inputs = torch.tensor[np.array]
        return encoded_inputs

image_processor = SegformerImageProcessor(do_reduce_labels=config['do_reduce_labels'])
inference_dataset = InferenceDataset(root_dir=config['root_dir'], image_processor=image_processor)
inference_dataloader = DataLoader(inference_dataset, batch_size=config['batch_size'], shuffle=False)

with open(config['id2label_path'], 'r') as f:
    id2label = json.load(f)
id2label = {int(k): v for k, v in id2label.items()}

model_name = config['model_name']
base_model_dir = os.path.join(config['base_dir'], model_name)
checkpoint_dir = os.path.join(base_model_dir, config['checkpoint_dir'])
checkpoint_csv = os.path.join(checkpoint_dir, 'checkpoints.csv')

def load_model(model_name, base_model_dir, config, checkpoint_dir):
    model_path = os.path.join(base_model_dir, model_name)
    if os.path.exists(checkpoint_csv):
        with open(checkpoint_csv, 'r') as f:
            reader = csv.DictReader(f)
            checkpoints = sorted(list(reader), key=lambda x: float(x['mean_iou']), reverse=True)
            if checkpoints:
                print("Available checkpoints:")
                for i, checkpoint in enumerate(checkpoints):
                    print(f"{i}: Epoch {checkpoint['epoch']}, Mean IoU: {checkpoint['mean_iou']}, Path: {checkpoint['path']}")
                choice = int(input("Enter the number of the checkpoint to load: "))
                checkpoint_path = checkpoints[choice]['path']
                print(f"Loading model from checkpoint {checkpoint_path}")
                model = SegformerForSemanticSegmentation.from_pretrained(
                    model_name,
                    num_labels=len(id2label),
                    id2label=id2label,
                    label2id={v: k for k, v in id2label.items()},
                    ignore_mismatched_sizes=True
                )
                state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys or unexpected_keys:
                    print(f"Missing keys: {missing_keys}")
                    print(f"Unexpected keys: {unexpected_keys}")
                return model
            else:
                raise FileNotFoundError("No checkpoint found. Ensure that the checkpoint directory is correct and contains valid checkpoints.")
    else:
        raise FileNotFoundError("No checkpoint CSV file found. Ensure that the checkpoint directory is correct and contains valid checkpoints.")

model = load_model(model_name, base_model_dir, config, checkpoint_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

flat_palette = []
for i in range(256):
    if i in priority_to_color:
        flat_palette.extend(priority_to_color[i])
    else:
        flat_palette.extend([0, 0, 0])

background_image_path = config['background_image_path']
background_image = Image.open(background_image_path).convert('L')
background_array = np.array(background_image)

with torch.no_grad():
    for batch in inference_dataloader:
        pixel_values = batch["pixel_values"].squeeze(1).to(device)
        image_name = batch['image_name'][0]
        ground_truth_annotation = batch['annotation'][0].cpu().numpy()
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(logits, size=(512, 896), mode="bilinear", align_corners=False)
        predicted = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
        np.save(os.path.join(segmentation_dir, f"{image_name.split('.')[0]}_label.npy"), predicted)
        softmax = torch.nn.functional.softmax(upsampled_logits, dim=1)
        confidence, _ = torch.max(softmax, dim=1)
        confidence = confidence.squeeze().cpu().numpy()
        confidence[background_array == 0] = 0
        segmentation_map = Image.fromarray(predicted.astype(np.uint8))
        segmentation_map.save(os.path.join(segmentation_dir, f"{image_name.split('.')[0]}_segmentation.png"))
        plt.imsave(os.path.join(heatmap_dir, f"{image_name.split('.')[0]}_heatmap.png"), confidence, cmap='hot')
        original_image = Image.open(os.path.join(inference_dataset.img_dir, image_name)).convert("RGBA")
        segmentation_overlay = Image.fromarray(predicted.astype(np.uint8), mode="P")
        segmentation_overlay.putpalette(flat_palette)
        segmentation_overlay = segmentation_overlay.convert("RGBA")
        overlay_image = Image.blend(original_image, segmentation_overlay, alpha=blend_alpha)
        overlay_image = np.array(overlay_image)
        overlay_image[background_array == 0] = 0
        overlay_image = Image.fromarray(overlay_image)
        overlay_image.save(os.path.join(overlay_dir, f"{image_name.split('.')[0]}_overlay.png"))
        ground_truth_annotation = Image.fromarray(ground_truth_annotation.astype(np.uint8)).resize((896, 512), resample=Image.NEAREST)
        ground_truth_overlay = Image.fromarray(np.array(ground_truth_annotation).astype(np.uint8), mode="P")
        ground_truth_overlay.putpalette(flat_palette)
        ground_truth_overlay = ground_truth_overlay.convert("RGBA")
        actual_overlay_image = Image.blend(original_image, ground_truth_overlay, alpha=blend_alpha)
        actual_overlay_image = np.array(actual_overlay_image)
        actual_overlay_image[background_array == 0] = 0
        actual_overlay_image = Image.fromarray(actual_overlay_image)
        actual_overlay_image.save(os.path.join(actual_dir, f"{image_name.split('.')[0]}_actual.png"))

print("Inference completed. Heatmaps, segmentation maps, and overlays are saved in the 'inference' directory.")
