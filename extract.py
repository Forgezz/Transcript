import requests
import chardet
import os
import re
import whisper
import torch
import gc
import subprocess
import transcript
from bs4 import BeautifulSoup
from pydub import AudioSegment
from urllib.parse import urlparse

#bilibili
def extract_audio_url_from_blbl(video_url, file_name, output_dir="downloads"):
    """
    Extract audio from a Bilibili video and provide HTML soup.
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 设置请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.bilibili.com/'  # 添加 Referer 头信息
    }

    # 获取网页内容
    try:
        response = requests.get(video_url, headers=headers)  # 添加 headers
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')  # 创建 soup 对象
    except requests.RequestException as e:
        raise Exception(f"获取网页内容失败: {e}")

    # yt-dlp 命令，提取音频资源
    command = [
        "yt-dlp", 
        "--extract-audio",                # 提取音频
        "--keep-video",                   # 保留原始音频格式
        "--output", f"{output_dir}/{file_name}.%(ext)s",  # 使用用户输入的文件名
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
                return latest_file, soup  # 返回与文件名匹配的 m4a 文件路径和 soup
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

if __name__ == "__main__":
    # 提示输入链接、文件名

    podcast_url = input("请输入链接（blbl/apple podcast/小宇宙）: ")
    file_name = input("请输出保存文件名: ")

    try:
        # 1. 提取音频 URL 和 HTML 内容
        audio_url, soup = extract_audio_url_by_platform(podcast_url, file_name)
        print(f"提取的音频 URL: {audio_url}")

        # 2. 下载音频文件
        downloaded_file = download_audio_file(audio_url, file_name)
        print(f"音频文件已下载: {downloaded_file}")

        # 3. 转换为 WAV 格式
        wav_file = convert_audio_to_wav(downloaded_file, file_name)
        print(f"音频文件已转换为 WAV 格式: {wav_file}")

        model = transcript.load_whisper_model_v3("large-v3")

        transcription_result = transcript.transcribe_audio_v3(model, wav_file)

        srt_file_path = transcript.save_transcription_as_srt_v3(transcription_result, wav_file)

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

