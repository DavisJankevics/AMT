import torch
import argparse
import os
from utils.utils import preprocess_audio, postprocess_output
from model.model import BiLSTM
from conf.conf import Config

def find_latest_checkpoint(checkpoint_dir):
    """
    Finds the latest checkpoint file in the specified directory.

    Args:
    - checkpoint_dir: Path to the directory containing checkpoint files.

    Returns:
    - The path to the latest checkpoint file.
    """
    checkpoint_files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    return latest_checkpoint

def load_checkpoint(load_path, model, device):
    """
    Loads model weights from a checkpoint file.

    Args:
    - load_path: Path to the checkpoint file.
    - model: The model to load weights into.
    - device: The device to perform computation on ('cuda' or 'cpu').
    """
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Model loaded from <== {load_path}')

def transcribe(audio_path, checkpoint_dir, device='cpu'):
    """
    Transcribes audio input from a file using the model loaded from the latest checkpoint.

    Args:
    - audio_path: Path to the audio file to transcribe.
    - checkpoint_dir: Directory containing model checkpoints.
    - device: The device to perform computation on ('cuda' or 'cpu').

    Returns:
    - transcription: The transcribed output.
    """

    config = Config()
    model = BiLSTM(config)
    
    # Identify and load the latest checkpoint
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    load_checkpoint(latest_checkpoint, model, device)
    model = model.to(device)
    model.eval()

    # Load and preprocess the audio file, then move it to the specified device
    audio = preprocess_audio(audio_path)  # Ensure preprocess_audio can load from file path
    audio = audio.to(device)
    print("\nhere\n")
    lengths = torch.tensor([audio.shape[0]], dtype=torch.long).to(device)

    with torch.no_grad():
        # Note: .unsqueeze(0) adds a batch dimension
        # 'lengths' is also expected to be a tensor containing the lengths of sequences
        output = model(audio.unsqueeze(0), lengths)  
    transcription = postprocess_output(output, device, "out.mid")  # Ensure postprocess_output handles device
    
    
    return transcription

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transcribe audio using a trained BiLSTM model')
    parser.add_argument('--audio_path', type=str, required=True, help='Path to the audio file to transcribe')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory containing model checkpoints')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    transcribe(args.audio_path, args.checkpoint_dir, device)
