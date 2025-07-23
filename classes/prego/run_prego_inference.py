#!/usr/bin/env python3
"""
PREGO Inference Script
This script runs the complete PREGO pipeline for online mistake detection in procedural egocentric videos.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import yaml

# Add PREGO to path
sys.path.append('PREGO')
sys.path.append('PREGO/step_recognition')
sys.path.append('PREGO/step_anticipation')


class PregoInference:
    """Main class for PREGO inference pipeline"""
    
    def __init__(self, config_path, checkpoint_path, dataset='Assembly101-O'):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.dataset = dataset
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize model components
        self.step_recognizer = None
        self.mistake_detector = None
        
    def load_step_recognition_model(self):
        """Load the MiniROAD step recognition model"""
        print("Loading Step Recognition Model (MiniROAD)...")
        
        # Import MiniROAD model
        from step_recognition.models.miniroad import MiniROAD
        
        # Initialize model
        num_classes = self.config['model']['num_classes']
        hidden_dim = self.config['model']['hidden_dim']
        num_layers = self.config['model']['num_layers']
        
        self.step_recognizer = MiniROAD(
            input_dim=2048,  # TSN feature dimension
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=self.config['model']['dropout']
        ).to(self.device)
        
        # Load checkpoint
        if os.path.exists(self.checkpoint_path):
            print(f"Loading checkpoint from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.step_recognizer.load_state_dict(checkpoint['model_state_dict'])
            self.step_recognizer.eval()
        else:
            print(f"Warning: Checkpoint not found at {self.checkpoint_path}")
            print("Using randomly initialized model")
            
    def process_video_features(self, video_path):
        """Process pre-extracted video features"""
        print(f"Processing video features: {video_path}")
        
        # Load pre-extracted TSN features
        features_rgb = np.load(f"{video_path}_rgb.npy")
        features_flow = np.load(f"{video_path}_flow.npy")
        
        # Combine RGB and flow features
        features = np.concatenate([features_rgb, features_flow], axis=1)
        
        return torch.tensor(features, dtype=torch.float32).to(self.device)
        
    def run_step_recognition(self, features):
        """Run online step recognition on video features"""
        print("Running step recognition...")
        
        predictions = []
        hidden = None
        
        with torch.no_grad():
            # Process features frame by frame (online)
            for t in range(features.shape[0]):
                frame_feature = features[t:t+1]
                
                # Run model
                if hasattr(self.step_recognizer, 'forward_step'):
                    output, hidden = self.step_recognizer.forward_step(frame_feature, hidden)
                else:
                    output = self.step_recognizer(frame_feature)
                
                # Get prediction
                pred = torch.argmax(output, dim=1).cpu().numpy()[0]
                predictions.append(pred)
                
        return predictions
        
    def aggregate_predictions(self, predictions, window_size=5):
        """Aggregate frame-level predictions"""
        print("Aggregating predictions...")
        
        aggregated = []
        current_action = None
        action_start = 0
        
        for i, pred in enumerate(predictions):
            # Simple majority voting in a sliding window
            if i >= window_size:
                window = predictions[i-window_size:i+1]
                majority_action = max(set(window), key=window.count)
                
                if majority_action != current_action:
                    if current_action is not None:
                        aggregated.append({
                            'action': current_action,
                            'start': action_start,
                            'end': i-1
                        })
                    current_action = majority_action
                    action_start = i
                    
        # Add final action
        if current_action is not None:
            aggregated.append({
                'action': current_action,
                'start': action_start,
                'end': len(predictions)-1
            })
            
        return aggregated
        
    def detect_mistakes(self, recognized_actions, expected_sequence=None):
        """Detect procedural mistakes based on recognized actions"""
        print("Detecting procedural mistakes...")
        
        mistakes = []
        
        # Simple rule-based mistake detection
        # In real implementation, this would use the LLAMA-based anticipation module
        for i in range(1, len(recognized_actions)):
            prev_action = recognized_actions[i-1]['action']
            curr_action = recognized_actions[i]['action']
            
            # Example: detect repeated actions (potential mistake)
            if prev_action == curr_action:
                mistakes.append({
                    'type': 'repeated_action',
                    'timestamp': recognized_actions[i]['start'],
                    'action': curr_action,
                    'confidence': 0.8
                })
                
            # Add more sophisticated mistake detection rules here
            
        return mistakes
        
    def run_inference(self, video_path, output_path):
        """Run complete PREGO inference pipeline"""
        print(f"Running PREGO inference on video: {video_path}")
        
        # Load model if not already loaded
        if self.step_recognizer is None:
            self.load_step_recognition_model()
            
        # Process video features
        features = self.process_video_features(video_path)
        
        # Run step recognition
        predictions = self.run_step_recognition(features)
        
        # Aggregate predictions
        recognized_actions = self.aggregate_predictions(predictions)
        
        # Detect mistakes
        mistakes = self.detect_mistakes(recognized_actions)
        
        # Prepare results
        results = {
            'video': video_path,
            'recognized_actions': recognized_actions,
            'detected_mistakes': mistakes,
            'total_frames': len(predictions),
            'total_mistakes': len(mistakes)
        }
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to: {output_path}")
        print(f"Total actions recognized: {len(recognized_actions)}")
        print(f"Total mistakes detected: {len(mistakes)}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='PREGO Inference Script')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to video features (without _rgb.npy or _flow.npy suffix)')
    parser.add_argument('--config', type=str, 
                        default='PREGO/step_recognition/configs/miniroad_assembly101-O.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, choices=['Assembly101-O', 'Epic-tent-O'],
                        default='Assembly101-O', help='Dataset to use')
    parser.add_argument('--output', type=str, default='results/inference_output.json',
                        help='Path to save results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize PREGO
    prego = PregoInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        dataset=args.dataset
    )
    
    # Run inference
    results = prego.run_inference(args.video, args.output)
    
    # Print summary
    print("\n=== Inference Complete ===")
    print(f"Video: {args.video}")
    print(f"Recognized Actions: {len(results['recognized_actions'])}")
    print(f"Detected Mistakes: {len(results['detected_mistakes'])}")
    
    if results['detected_mistakes']:
        print("\nMistakes found:")
        for i, mistake in enumerate(results['detected_mistakes']):
            print(f"  {i+1}. {mistake['type']} at frame {mistake['timestamp']}")


if __name__ == "__main__":
    main()