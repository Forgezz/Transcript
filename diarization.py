# diarization.py
from pyannote.audio import Pipeline
import srt
import datetime
import torch  # 导入torch模块


def diarize_audio(audio_path):
    """使用 pyannote.audio 进行说话人分离"""
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="TOKEN")
        pipeline = pipeline.to(torch.device('cuda:0'))
        
        diarization = pipeline(audio_path)
        return diarization
    except Exception as e:
        print(f"说话人分离失败: {e}")
        return None  # 处理异常情况

def process_srt_with_diarization(srt_file, diarization):
    """将说话人分离结果与 SRT 文件合并"""
    if diarization is None:
        return None  # 如果说话人分离失败，则返回 None

    segments = list(srt.parse(open(srt_file, "r", encoding="utf-8")))
    diarized_segments = []
    speaker_labels = {}  # 用于存储说话人标签

    for segment in segments:
        start_time = segment.start.total_seconds()
        end_time = segment.end.total_seconds()

        # 找到与当前字幕片段重叠的说话人片段
        overlapping_speakers = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.start <= end_time and turn.end >= start_time:
                overlapping_speakers.append((turn, speaker))

        if overlapping_speakers:
            # 处理多个说话人重叠的情况，这里选择重叠时间最长的说话人
            best_turn, speaker = max(overlapping_speakers, key=lambda x: min(end_time, x[0].end) - max(start_time, x[0].start))
            if speaker not in speaker_labels:
                speaker_labels[speaker] = f"发言者{len(speaker_labels) + 1}"  # 生成新的说话人标签
            diarized_segments.append({
                "start": start_time,
                "end": end_time,
                "text": segment.content,
                "speaker": speaker_labels[speaker]
            })
        else:
            diarized_segments.append({
                "start":start_time,
                "end":end_time,
                "text":segment.content,
                "speaker":"未知说话人"
            })
    return diarized_segments

def save_diarized_transcription(diarized_segments, output_file):
    """保存带有说话人标签的转录文本"""
    if diarized_segments is None:
        return  # 如果处理后的片段为空，则直接返回

    with open(output_file, "w", encoding="utf-8") as f:
        for segment in diarized_segments:
            f.write(f"{segment['speaker']}：{segment['text']}\n")