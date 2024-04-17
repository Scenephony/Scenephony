import os
import subprocess
from tqdm import tqdm


def download(name: str, url: str, save_dir: str = "videos"):
    """Download the 480p video by calling youtube-dl, save to disk.
    
    Args:
        name: name of the video.
        url: youtube url of the video.
        save_dir: directory to save the video.
    """
    os.makedirs(save_dir, exist_ok=True)
    opts = ['-f', 'bestvideo[height<=360]+bestaudio/best[height<=480]', url, \
            '--merge-output-format', 'mp4', '-o', f'{save_dir}/{name}.%(ext)s']
    subprocess.run(['youtube-dl', *opts], stdout=subprocess.DEVNULL)


def download_csv(csv_path: str, start: int, end: int, save_dir: str = "videos"):
    """Download videos from a csv file.
    
    Args:
        csv_path: path to the csv file, each line has format: name, youtube_url, ...
        start: start index to download.
        end: end index to download.
        save_dir: directory to save the videos.
    """
    with open(csv_path, 'r') as f:
        lines = f.readlines()[1:]
        
        # Download each video
        for i in tqdm(range(start, min(end, len(lines)))):
            name, url = lines[i].strip().split()[:2]
            download(name, url, save_dir)


if __name__ == '__main__':
    import sys
    start_i, end_i = int(sys.argv[1]), int(sys.argv[2])
    download_csv('SymMV.csv', start_i, end_i)