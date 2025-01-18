import whisper
import gc
import os

def load_whisper_model_v3(model_name="large-v3"):
    """
    Load the Whisper model (large-v3) from Hugging Face.
    """
    print(f"Loading Whisper model: {model_name}...")
    model = whisper.load_model(model_name)
    print(f"Model {model_name} loaded successfully.")
    return model


def transcribe_audio_v3(model, audio_path, task="transcribe", language=None,verbose=False):
    """
    Transcribe audio using Whisper model (large-v3).
    """
    print(f"Transcribing audio: {audio_path}...")
    result = model.transcribe(audio_path, task=task, language=language, verbose=verbose)#增加了最后一个verbose
    print("Transcription completed.")
    return result


def save_transcription_as_srt_v3(transcription, audio_path):
    """
    Save transcription result as SRT file.
    """
    srt_file_path = os.path.splitext(audio_path)[0] + ".srt"
    with open(srt_file_path, "w", encoding="utf-8") as srt_file:
        for segment in transcription['segments']:
            start = segment['start']
            end = segment['end']
            text = segment['text']
            # Write to SRT format
            srt_file.write(f"{segment['id'] + 1}\n")
            srt_file.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            srt_file.write(f"{text}\n\n")
    print(f"Transcription saved as SRT: {srt_file_path}")
    return srt_file_path


def format_timestamp(seconds):
    """
    Format timestamp for SRT file.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"


