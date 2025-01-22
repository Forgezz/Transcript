# extract.py
import requests
import chardet
import os
import re
import whisper
import torch
import gc
import subprocess
import transcript
import diarization # 导入 diarization 模块
from bs4 import BeautifulSoup
from pydub import AudioSegment
from urllib.parse import urlparse

#bilibili and Youtube
def extract_audio_url_from_blbl(video_url, file_name, output_dir="downloads"):
    """
    Extract audio from a Bilibili or YouTube video using yt-dlp.
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # yt-dlp 命令，提取音频资源

    command = [
    "yt-dlp",
    "--extract-audio",
    "--audio-format", "m4a",
    "--no-check-certificate",  # 添加此参数
    "--output", f"{output_dir}/{file_name}.%(ext)s",
    video_url
]

    try:
        # 使用 subprocess 运行命令
        subprocess.run(command, check=True)

        # 获取生成的文件名
        output_files = os.listdir(output_dir)
        if output_files:
            # 查找与指定文件名匹配的 .m4a 文件
            m4a_file = next((f for f in output_files if f.startswith(file_name) and f.endswith(".m4a")), None)
            if m4a_file:
                latest_file = os.path.join(output_dir, m4a_file)
                print(f"音频提取完成，文件保存在: {latest_file}")
                return latest_file, None  # 返回与文件名匹配的 m4a 文件路径
            else:
                raise Exception(f"未找到与 {file_name} 匹配的 .m4a 文件。")
        else:
            raise Exception("音频提取失败，输出目录为空。")
    except subprocess.CalledProcessError as e:
        raise Exception(f"提取音频时发生错误: {e}")



#ApplePodcast
def extract_audio_url_from_apple(url):
    """
    Extract audio URL and soup from an Apple Podcast page.
    """
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch the webpage. Status code: {response.status_code}\nURL: {url}")

    # 自动检测编码
    detected_encoding = chardet.detect(response.content)['encoding']
    if detected_encoding:
        response.encoding = detected_encoding
    else:
        response.encoding = 'utf-8'

    soup = BeautifulSoup(response.text, 'html.parser')

    # 查找音频链接 (MP3 或 M4A)
    audio_urls = re.findall(r'https://[^\s^"]+(?:\.mp3|\.m4a)', response.text)
    if not audio_urls:
        raise Exception("No audio URLs found on the page.")

    return audio_urls[-1], soup  # 返回音频链接和 HTML 结构

#小宇宙
def extract_audio_url_from_xiaoyuzhou(url):
    """
    Extract audio URL and soup from a Xiaoyuzhou FM page.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')

    # 查找音频链接 (og:audio)
    audio_meta = soup.find('meta', property='og:audio')
    if audio_meta and audio_meta.get('content'):
        return audio_meta['content'], soup  # 返回音频链接和 HTML 结构
    else:
        raise Exception("No audio URL found on the Xiaoyuzhou page.")

#判断函数
def extract_audio_url_by_platform(url, file_name):
    """
    Determine platform and extract audio URL and soup.
    """
    if "apple.com" in url:
        print("检测到 Apple Podcast 链接，开始提取...")
        return extract_audio_url_from_apple(url)
    elif "xiaoyuzhoufm.com" in url:
        print("检测到小宇宙链接，开始提取...")
        return extract_audio_url_from_xiaoyuzhou(url)
    elif "bilibili.com" in url:
        print("检测到哔哩哔哩链接，开始提取...")
        return extract_audio_url_from_blbl(url, file_name) 

    elif "youtube.com" in url or "youtu.be" in url:
        print("检测到Youtube链接，开始提取...")
        return extract_audio_url_from_blbl(url, file_name) 
    
    else:
        raise Exception("无法识别的链接平台，请提供 Apple Podcast、小宇宙或 Bilibili 平台的链接。")

def download_audio_file(audio_url, title, output_folder="downloads"):
    """
    Download the audio file from a given URL and save it locally with the given title.
    """
    # 检查 audio_url 是否是本地路径
    if os.path.exists(audio_url):
        # 如果是本地文件路径，直接返回
        print(f"音频文件已存在于本地: {audio_url}")
        return audio_url

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 格式化标题为文件名
    formatted_title = title
    file_extension = os.path.splitext(urlparse(audio_url).path)[1]
    output_path = os.path.join(output_folder, f"{formatted_title}{file_extension}")

    # Stream and save the音频 file
    with requests.get(audio_url, stream=True) as response:
        response.raise_for_status()
        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

    return output_path


def convert_audio_to_wav(input_file, title):
    """
    Convert audio file (MP3/M4A) to WAV format and save it with the given title.
    """
    formatted_title = title
    output_file = os.path.join(os.path.dirname(input_file), f"{formatted_title}.wav")
    
    if input_file.lower().endswith(".mp3"):
        audio = AudioSegment.from_mp3(input_file)
    elif input_file.lower().endswith(".m4a"):
        audio = AudioSegment.from_file(input_file, "m4a")
    else:
        raise Exception("Unsupported audio format. Only MP3 and M4A are supported.")
    
    audio.export(output_file, format="wav")
    return output_file

def trim_wav_start(input_file, output_file, trim_duration_ms=10000):
    """
    剪切 WAV 文件开头指定时长的音频。
    """
    try:
        audio = AudioSegment.from_wav(input_file)
        trimmed_audio = audio[trim_duration_ms:]
        trimmed_audio.export(output_file, format="wav")
        print(f"已将 {input_file} 剪切为 {output_file}")
        return output_file
    except Exception as e:
        print(f"剪切 WAV 文件时发生错误: {e}")
        return None
if __name__ == "__main__":
    # 提示输入链接、文件名
    podcast_url = input("请输入链接（blbl/apple podcast/小宇宙）: ")
    file_name = input("请输出保存文件名: ")

    try:
        # 提取音频 URL 和 HTML 内容
        audio_url, soup = extract_audio_url_by_platform(podcast_url, file_name)
        print(f"提取的音频 URL: {audio_url}")

        # 下载音频文件
        downloaded_file = download_audio_file(audio_url, file_name)
        print(f"音频文件已下载: {downloaded_file}")

        # 转换为 WAV 格式
        wav_file = convert_audio_to_wav(downloaded_file, file_name)
        print(f"音频文件已转换为 WAV 格式: {wav_file}")

        # 询问是否需要裁剪以及裁剪时长
        while True:
            try:
                trim_duration = int(input("请输入需要裁剪的秒数（输入0则不裁剪）: "))
                if trim_duration >= 0:
                    break
                else:
                    print("请输入非负整数。")
            except ValueError:
                print("请输入有效的整数。")

        if trim_duration > 0:
            # 剪切音频文件
            trimmed_wav_file = os.path.splitext(wav_file)[0] + "_trimmed.wav"
            trimmed_wav_file = trim_wav_start(wav_file, trimmed_wav_file, trim_duration * 1000) # trim_duration * 1000 转换为毫秒
            if trimmed_wav_file is None:
                print("音频剪切失败，使用原文件进行转录。")
                trimmed_wav_file = wav_file
            else:
                print(f"已创建剪切后的音频文件: {trimmed_wav_file}")
            audio_file_to_process = trimmed_wav_file #将需要处理的文件指向裁剪后的文件
        else:
            print("选择不裁剪音频。")
            audio_file_to_process = wav_file #不裁剪，使用原文件

        # 使用处理后的音频文件进行转录
        model = transcript.load_whisper_model_v3("large-v3")
        transcription_result = transcript.transcribe_audio_v3(model, audio_file_to_process)
        srt_file_path = transcript.save_transcription_as_srt_v3(transcription_result, audio_file_to_process)
        print(f"转录完成！字幕文件保存在: {srt_file_path}")

        # 保存为 TXT 文件
        txt_file_path = transcript.save_transcription_as_txt_v3(transcription_result, audio_file_to_process)
        print(f"转录文本保存为 TXT: {txt_file_path}")

        # 4. 进行说话人分离，使用处理后的文件
        print("Performing speaker diarization...")
        diarization_result = diarization.diarize_audio(audio_file_to_process)
        print("Speaker diarization completed.")

        if diarization_result is None:
            raise Exception("说话人分离失败，无法继续处理。")

        # 5. 处理 SRT 文件并添加说话人标签
        print("Processing SRT file with diarization results...")
        diarized_segments = diarization.process_srt_with_diarization(srt_file_path, diarization_result)

        if diarized_segments is None:
            raise Exception("SRT 文件处理失败，无法继续处理。")

        # 6. 保存带有说话人标签的转录文本
        output_txt_file = os.path.splitext(audio_file_to_process)[0] + "_diarized.txt" #使用处理后的文件名
        diarization.save_diarized_transcription(diarized_segments, output_txt_file)
        print(f"Diarized transcription saved to: {output_txt_file}")

    except Exception as e:
        print(f"处理过程中发生错误: {e}")

    finally:
        # 显存释放
        print("Releasing GPU memory...")
        if 'model' in locals(): #添加判断，防止model未定义时报错
            del model  # 删除模型对象
        torch.cuda.empty_cache()  # 释放未使用的显存
        gc.collect()  # 强制进行垃圾回收
        print("GPU memory released.")
        
