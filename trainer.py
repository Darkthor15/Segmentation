import os
import json
import sys
import mlflow
import warnings
from urllib.parse import urlparse
import logging
import mlflow.pytorch  # You can use this if you want to log your PyTorch model
import torch
import shutil
import random
import csv
import evaluate
import pandas as pd
from torch import nn
from tqdm import tqdm
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from torch.utils.data import Dataset, DataLoader
from mlflow.models import infer_signature

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)


mlflow.set_tracking_uri('http://0.0.0.0:5000') #set tracking uri
experiment = mlflow.set_experiment("segmentation-model") #set experiment name

# Create directories for model artifacts and checkpoints
base_model_dir = os.path.join(config['base_dir'], config['model_name'])
os.makedirs(base_model_dir, exist_ok=True)
checkpoint_dir = os.path.join(base_model_dir, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_csv = os.path.join(checkpoint_dir, 'checkpoints.csv')

class SemanticSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_processor, train=True):
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.train = train
        sub_path = "training" if self.train else "validation"
        self.img_dir = os.path.join(self.root_dir, "images", sub_path)
        self.ann_dir = os.path.join(self.root_dir, "annotations", sub_path)
        self.images = sorted(os.listdir(self.img_dir))
        self.annotations = sorted(os.listdir(self.ann_dir))
        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.img_dir, self.images[idx])).convert("RGB")
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))
        image = image.resize((512, 896))
        segmentation_map = segmentation_map.resize((512, 896), resample=Image.NEAREST)
        encoded_inputs = self.image_processor(image, segmentation_map, return_tensors="pt")
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()
        return encoded_inputs

image_processor = SegformerImageProcessor(do_reduce_labels=config['do_reduce_labels'], size=(512, 896))
train_dataset = SemanticSegmentationDataset(root_dir=config['root_dir'], image_processor=image_processor)
valid_dataset = SemanticSegmentationDataset(root_dir=config['root_dir'], image_processor=image_processor, train=False)
train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=config['batch_size'])

id2label_path = "lawnmower_label.json"
with open(id2label_path, 'r') as f:
    id2label = json.load(f)
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

def load_base_model(model_name, base_model_dir, config):
    model_path = os.path.join(base_model_dir, model_name)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_path,
            num_labels=config['num_labels'],
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
    else:
        print(f"Downloading model from Hugging Face Hub and saving to {model_path}")
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=config['num_labels'],
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        model.save_pretrained(model_path)
    return model

def load_checkpoint(model, checkpoint_csv):
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
            state_dict = torch.load(checkpoint_path)
            model.load_state_dict(state_dict, strict=False)
        else:
            print("No checkpoint found. Training from scratch.")
    else:
        print("No checkpoint CSV file found. Training from scratch.")

def load_external_weights(model, config):
    weights_file_path = config["weights_file_path"]
    num_layers_to_load = config.get("num_layers_to_load", None)
    state_dict = torch.load(weights_file_path)
    if num_layers_to_load is not None:
        filtered_state_dict = {k: v for i, (k, v) in enumerate(state_dict.items()) if i < num_layers_to_load}
        model.load_state_dict(filtered_state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)

def freeze_model_layers(model, config):
    num_layers_to_freeze = config.get("num_layers_to_freeze", 0)
    layers = list(model.named_parameters())
    for name, param in layers[:num_layers_to_freeze]:
        param.requires_grad = False
        print(f"Layer {name} frozen.")

def load_model(model_name, base_model_dir, config, checkpoint_csv, from_checkpoint):
    model = load_base_model(model_name, base_model_dir, config)
    if from_checkpoint:
        load_checkpoint(model, checkpoint_csv)
    if config.get("load_weights_from_file", False):
        load_external_weights(model, config)
    if config.get("freeze_layers", False):
        freeze_model_layers(model, config)
    return model

from_checkpoint = config.get('from_checkpoint', False)
model = load_model(config['model_name'], base_model_dir, config, checkpoint_csv, from_checkpoint)
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
metric = evaluate.load("mean_iou")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)

def save_model(model, path):
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), path)

def save_checkpoint_metadata(epoch, val_metrics, path):
    fieldnames = ['epoch', 'mean_iou', 'mean_accuracy', 'path']
    if not os.path.exists(checkpoint_csv):
        with open(checkpoint_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({'epoch': epoch, 'mean_iou': val_metrics['mean_iou'], 'mean_accuracy': val_metrics['mean_accuracy'], 'path': path})
    else:
        with open(checkpoint_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({'epoch': epoch, 'mean_iou': val_metrics['mean_iou'], 'mean_accuracy': val_metrics['mean_accuracy'], 'path': path})

def manage_checkpoints(checkpoint_dir, num_checkpoints):
    if os.path.exists(checkpoint_csv):
        with open(checkpoint_csv, 'r') as f:
            reader = csv.DictReader(f)
            checkpoints = sorted(list(reader), key=lambda x: float(x['mean_iou']), reverse=True)
        while len(checkpoints) > num_checkpoints:
            to_remove = checkpoints.pop(-1)
            os.remove(to_remove['path'])
        with open(checkpoint_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'mean_iou', 'mean_accuracy', 'path'])
            writer.writeheader()
            for checkpoint in checkpoints:
                writer.writerow(checkpoint)
    else:
        print("No checkpoint CSV file found for managing checkpoints.")

class WeightedCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, weights=None, ignore_index=-100):
        super().__init__(weight=weights, ignore_index=ignore_index)
    def forward(self, input, target):
        return super().forward(input, target)

class_weights = torch.tensor(config.get('class_weights', [1.0] * config['num_labels']), device=device)
criterion = WeightedCrossEntropyLoss(weights=class_weights, ignore_index=config['ignore_index'])

with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
    mlflow.log_params(config)
    
    model.train()
    losses = []
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch: {epoch + 1}/{config['num_epochs']}\n" + "-" * 50)
        running_loss = 0
        for idx, batch in enumerate(tqdm(train_dataloader)):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            dummy_input = {"pixel_values": pixel_values.cpu().numpy(),"labels": labels.cpu().numpy()}
            outputs = model(pixel_values=pixel_values, labels=labels)
            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(logits, size=(512, 896), mode="bilinear", align_corners=False)
            loss = criterion(upsampled_logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            with torch.no_grad():
                predicted = upsampled_logits.argmax(dim=1)
                metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
            if idx % 100 == 0:
                model.eval()
                val_loss = 0
                val_steps = 0
                val_metric = evaluate.load("mean_iou")
                for val_batch in valid_dataloader:
                    val_pixel_values = val_batch["pixel_values"].to(device)
                    val_labels = val_batch["labels"].to(device)
                    with torch.no_grad():
                        val_outputs = model(pixel_values=val_pixel_values, labels=val_labels)
                        val_logits = val_outputs.logits
                        upsampled_val_logits = nn.functional.interpolate(val_logits, size=(512, 896), mode="bilinear", align_corners=False)
                        val_loss += criterion(upsampled_val_logits, val_labels).item()
                        val_predicted = upsampled_val_logits.argmax(dim=1)
                        val_metric.add_batch(predictions=val_predicted.detach().cpu().numpy(), references=val_labels.detach().cpu().numpy())
                    val_steps += 1
                val_loss /= val_steps
                val_metrics = val_metric.compute(num_labels=config['num_labels'], ignore_index=config['ignore_index'])
                tqdm.write(f"Validation - Loss: {val_loss:.4f}, Mean IoU: {val_metrics['mean_iou']:.4f}, Mean Accuracy: {val_metrics['mean_accuracy']:.4f}")
                model.train()
        avg_training_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} - Average Training Loss: {avg_training_loss:.4f}\n")
        losses.append({
            'epoch': epoch + 1,
            'average_training_loss': avg_training_loss,
            'validation_loss': val_loss,
            'mean_iou': val_metrics['mean_iou'],
            'mean_accuracy': val_metrics['mean_accuracy']
        })

        # Log metrics for each epoch in MLflow
        mlflow.log_metric("train_loss", avg_training_loss, step=epoch + 1)
        mlflow.log_metric("val_loss", val_loss, step=epoch + 1)
        mlflow.log_metric("mean_iou", val_metrics['mean_iou'], step=epoch + 1)
        mlflow.log_metric("mean_accuracy", val_metrics['mean_accuracy'], step=epoch + 1)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pt")
        save_model(model, checkpoint_path)
        save_checkpoint_metadata(epoch + 1, val_metrics, checkpoint_path)
        mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")  #log artifact as well
        manage_checkpoints(checkpoint_dir, config['num_checkpoints'])
        signature = infer_signature(dummy_input , logits.detach().cpu().numpy())
    

    mlflow.set_tag("Training Info", "Everything about segmentation")
    # Save the training losses log file as an artifact
    with open(config['log_file'], 'w') as f:
        json.dump(losses, f, indent=4)
    mlflow.log_artifact(config['log_file'], artifact_path="logs")

    mlflow.pytorch.log_model(model, artifact_path="pytorch_model" , input_example = dummy_input , signature = signature  , registered_model_name = 'segmentation-model') #log model as well
    
warnings.filterwarnings("ignore", message="Downcasting array dtype")
