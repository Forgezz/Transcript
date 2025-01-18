import whisper
import gc

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



# 主程序：模块二集成
if __name__ == "__main__":
    # 假设模块一已经完成，直接获取 WAV 文件路径
    wav_file_path = wav_file  # 上一模块生成的结果

    # 确保文件存在
    if not wav_file_path or not os.path.exists(wav_file_path):
        print(f"错误：音频文件 {wav_file_path} 不存在！请确保模块一已正确运行。")
    else:
        try:
            # 1. 加载 Whisper large-v3 模型
            model = load_whisper_model_v3("large-v3")

            # 2. 转录音频
            transcription_result = transcribe_audio_v3(model, wav_file_path)

            # 3. 保存转录结果为 SRT 文件
            srt_file_path = save_transcription_as_srt_v3(transcription_result, wav_file_path)

            print(f"转录完成！字幕文件保存在: {srt_file_path}")

        except Exception as e:
            print(f"处理过程中发生错误: {e}")

        finally:
            # 显存释放
            print("Releasing GPU memory...")
            del model  # 删除模型对象
            torch.cuda.empty_cache()  # 释放未使用的显存
            gc.collect()  # 强制进行垃圾回收
            print("GPU memory released.")