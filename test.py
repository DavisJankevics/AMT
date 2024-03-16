import os
import argparse
from omnizart.music import app as music_app
import subprocess

def transcribe_directory(audio_dir, output_dir, mode):
    """
    Transcribes all audio files in the given directory and saves the transcriptions
    to the specified output directory.

    Parameters:
    - audio_dir: Directory containing audio files to transcribe.
    - output_dir: Directory where the MIDI files will be saved.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the pre-trained music model
    # music_app.load_model("music")

    # Iterate over all files in the directory
    for file_name in os.listdir(audio_dir):
        file_path = os.path.join(audio_dir, file_name)
        if os.path.isfile(file_path):
            print(f"Transcribing: {file_path}")
            try:
                # Perform transcription and save the output
                output_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + ".mid")
                # music_app.transcribe_music(file_path, output=output_path)
                subprocess.run(f"omnizart music transcribe {file_path} -o ./output/{mode}/omnizart", shell=True)
                
                print(f"Saved transcription to: {output_path}")
            except Exception as e:
                print(f"Error transcribing {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files to MIDI.")
    parser.add_argument("--audio_dir", type=str, help="Directory containing audio files to transcribe.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the MIDI transcriptions.")
    parser.add_argument("--mode", type=str, help="baseline or mss")
    args = parser.parse_args()
    transcribe_directory(args.audio_dir, args.output_dir, args.mode)

if __name__ == "__main__":
    main()
