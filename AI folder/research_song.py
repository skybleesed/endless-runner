#!/usr/bin/env python3
import sys, json, os, tempfile
import yt_dlp
import librosa
import soundfile as sf
import subprocess
import glob

def get_ffmpeg_path():
    """Return the directory containing ffmpeg.exe if found, else None."""
    base = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Microsoft\WinGet\Packages')
    if not os.path.isdir(base):
        return None
    for root, dirs, files in os.walk(base):
        if 'ffmpeg.exe' in files:
            return root
    return None

FFMPEG_PATH = get_ffmpeg_path()

def get_metadata(url):
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,   # Không tải video
        'skip_download': True,
    }
    if FFMPEG_PATH:
        ydl_opts['ffmpeg_location'] = FFMPEG_PATH
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if 'entries' in info:  # Playlist → lấy video đầu
            info = info['entries'][0]
        title = info.get('title', 'Unknown')
        uploader = info.get('uploader', info.get('artist', 'Unknown'))
        # Các trường hữu ích
        duration = info.get('duration', 0)
        tags = info.get('tags', [])
        categories = info.get('categories', [])
        # Thể loại (genre) nếu có
        genre = info.get('genre', ' / '.join(categories + tags[:2])) or 'Unknown'
        return {
            'title': title,
            'artist': uploader,
            'duration': duration,
            'genre': genre,
            'bpm': None,  # sẽ ước lượng sau
            'energy': None,
            'valence': None,
        }

def download_audio_segment(url, duration=30):
    """Tải một đoạn audio ngắn, trả về đường dẫn file tạm và thư mục tạm"""
    tmpdir = tempfile.mkdtemp()
    outtmpl = os.path.join(tmpdir, 'audio.%(ext)s')
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': outtmpl,
        'quiet': True,
        'no_warnings': True,
        'playlist_items': '1',
        'download_sections': f'*0:00-0:{duration:02d}',  # cắt đúng 30s đầu
    }
    if FFMPEG_PATH:
        ydl_opts['ffmpeg_location'] = FFMPEG_PATH
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        # If download fails, clean up and return None
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass
        return None, None
    # Tìm file đã tải
    files = []
    for f in os.listdir(tmpdir):
        if os.path.isfile(os.path.join(tmpdir, f)):
            files.append(f)
    if not files:
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass
        return None, None
    # Giả sử chỉ có một file (audio)
    audio_path = os.path.join(tmpdir, files[0])
    return audio_path, tmpdir

def estimate_tempo(filepath):
    y, sr = librosa.load(filepath, duration=30)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return int(round(tempo))

def estimate_energy_valence(filepath):
    # Rất thô sơ: dùng RMS làm đại diện energy, tính trung bình spectral centroid làm valence giả
    y, sr = librosa.load(filepath, duration=30)
    rms = librosa.feature.rms(y=y).mean()
    # Scale rms lên khoảng 0-1 (tương đối)
    energy = min(1.0, rms * 10)  # hệ số tùy chỉnh
    spectral = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    # Valence giả: map centroid (thường cao → tươi sáng)
    valence = min(1.0, spectral / 5000)
    return round(energy, 3), round(valence, 3)

def main():
    if len(sys.argv) < 2:
        print("Usage: python research_song.py <url>")
        sys.exit(1)
    url = sys.argv[1]
    print(f"🔍 Đang phân tích: {url}", file=sys.stderr)

    # Bước 1: metadata
    meta = get_metadata(url)
    print(f"   Tên: {meta['title']} - {meta['artist']}", file=sys.stderr)

    # Bước 2: Tải audio tạm để ước lượng BPM, energy, valence
    result = download_audio_segment(url, duration=30)
    if result[0] is not None:
        audio_path, tmpdir = result
        try:
            meta['bpm'] = estimate_tempo(audio_path)
            meta['energy'], meta['valence'] = estimate_energy_valence(audio_path)
        except Exception as e:
            print(f"   ⚠️ Lỗi xử lý âm thanh: {e}", file=sys.stderr)
        finally:
            # Xóa file tạm và thư mục
            try:
                os.unlink(audio_path)
            except OSError:
                pass
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass
    else:
        print("   ⚠️ Không tải được audio để phân tích BPM.", file=sys.stderr)

    # Xuất JSON kết quả
    print(json.dumps(meta, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
