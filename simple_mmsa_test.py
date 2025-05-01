#!/usr/bin/env python3
"""
Simplified script to test the Self_MM model for multimodal sentiment analysis.
This version saves the output to a file for easier inspection.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
import json
import argparse
import traceback

# Add the project root to the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

class AttrDict(dict):
    """A dictionary that allows for attribute-style access while preserving dict-style access."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
    
    def get(self, key, default=None):
        return self[key] if key in self else default

def main():
    parser = argparse.ArgumentParser(description='Test Self_MM model for multimodal sentiment analysis')
    parser.add_argument('--video_path', type=str, default='./test_videos/Calm.mp4', help='Path to a video file')
    parser.add_argument('--model_path', type=str, default='./pretrained_models/self_mm-mosi_fixed.pth', help='Path to model weights')
    parser.add_argument('--config_path', type=str, default='./pretrained_models/self_mm-mosi-config_fixed.json', help='Path to config file')
    args = parser.parse_args()
    
    # Create output log file
    log_file = open("test_results.txt", "w")
    
    # Helper function to both log and print
    def log_print(message):
        print(message)
        log_file.write(message + "\n")
        log_file.flush()
    
    # Load config
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    log_print(f"Loaded config from {args.config_path}")
    log_print(f"Model name: {config['model_name']}")
    
    try:
        from repositories.MMSA.src.MMSA.models import AMIO
        log_print("Successfully imported AMIO model")
        
        # Create feature directory
        Path("mmsa_fet_outputs").mkdir(parents=True, exist_ok=True)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_print(f"Using device: {device}")
        
        # Convert config dict to an AttrDict that supports dict-style access
        config_obj = AttrDict(config)
        
        # Initialize the model
        model = AMIO(config_obj).to(device)
        log_print("Model initialized")
        
        # Create dummy features
        from test_self_mm import create_dummy_features, score_to_label
        video_name = Path(args.video_path).stem
        log_print(f"Creating features for video: {video_name}")
        features = create_dummy_features(args.video_path, "mmsa_fet_outputs", True)
        
        # Convert features to tensors
        audio_tensor = torch.FloatTensor(features['audio']).unsqueeze(0).to(device)
        vision_tensor = torch.FloatTensor(features['vision']).unsqueeze(0).to(device)
        
        # Create sequence lengths
        seq_len = features['text'].shape[0]
        text_lengths = torch.tensor([seq_len]).to(device)
        audio_lengths = torch.tensor([features['audio'].shape[0]]).to(device)
        vision_lengths = torch.tensor([features['vision'].shape[0]]).to(device)
        
        # Create a simulated BERT input
        batch_size = 1
        text_input = torch.zeros(batch_size, 3, seq_len, device=device)
        
        # Create random token IDs
        random_token_ids = torch.randint(0, 100, (batch_size, seq_len), device=device).long()
        text_input[:, 0, :] = random_token_ids  # input_ids
        text_input[:, 1, :] = 1.0  # attention_mask
        text_input[:, 2, :] = 0  # token_type_ids
        
        # Package audio and vision features
        audio_input = (audio_tensor, audio_lengths)
        vision_input = (vision_tensor, vision_lengths)
        
        # Load model weights
        try:
            log_print(f"Loading model weights from {args.model_path}")
            state_dict = torch.load(args.model_path, map_location=device)
            
            # Handle potential key mismatches
            log_print("Loading with non-strict mode to handle mismatches")
            model_state_dict = model.state_dict()
            matched_keys = 0
            
            for k, v in state_dict.items():
                if k in model_state_dict and v.shape == model_state_dict[k].shape:
                    model_state_dict[k] = v
                    matched_keys += 1
            
            # Load filtered state dict
            model.load_state_dict(model_state_dict, strict=False)
            log_print(f"Successfully loaded {matched_keys} matching parameters")
            
        except Exception as e:
            log_print(f"Error loading model weights: {str(e)}")
            traceback.print_exc(file=log_file)
            sys.exit(1)
        
        # Set model to evaluation mode
        model.eval()
        
        # Run inference
        with torch.no_grad():
            log_print("Running inference...")
            try:
                outputs = model(text_input, audio_input, vision_input, video_name=video_name, print_modality_shapes=True)
                
                # Process outputs
                if isinstance(outputs, dict):
                    # Self_MM returns a dictionary with modality-specific predictions
                    prediction = outputs['M'].cpu().numpy()[0][0]  # Get the multimodal fusion result
                    text_pred = outputs['T'].cpu().numpy()[0][0]
                    audio_pred = outputs['A'].cpu().numpy()[0][0]
                    video_pred = outputs['V'].cpu().numpy()[0][0]
                    
                    log_print(f"\nMultimodal score: {prediction:.4f} ({score_to_label(prediction)})")
                    log_print(f"Text-only score: {text_pred:.4f} ({score_to_label(text_pred)})")
                    log_print(f"Audio-only score: {audio_pred:.4f} ({score_to_label(audio_pred)})")
                    log_print(f"Video-only score: {video_pred:.4f} ({score_to_label(video_pred)})")
                    
                    log_print("\nSelf-MM test completed successfully!")
            except Exception as e:
                log_print(f"Error during inference: {str(e)}")
                traceback.print_exc(file=log_file)
                
    except Exception as e:
        log_print(f"Error during execution: {str(e)}")
        traceback.print_exc(file=log_file)
    
    log_file.close()
    print("Test completed. Results saved to test_results.txt")

if __name__ == "__main__":
    main() 