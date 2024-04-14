import os
import torch
import torchvision
import torchaudio
from tqdm import tqdm
from typing import Optional

from model import ScenephonyModel


@torch.no_grad()
def test(
    data_dir: str,
    output_dir: str,
    seed: int = 42,
    checkpoint_path: Optional[str] = None,
):
    """Generate music using the model (trained if checkpoint_path is provided).

    Args:
        data_dir: Path to the test dataset.
        output_dir: Path to save the generated music.
        seed: Random seed.
        checkpoint_path: Path to the trained model checkpoint.
    """
    # Initial setup
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup/load model
    model = ScenephonyModel()
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    # Load data, inference and save audio files
    for vid_fn in tqdm(os.listdir(data_dir)):
        vid_path = os.path.join(data_dir, vid_fn)
        video = torchvision.io.read_video(vid_path, output_format="TCHW")[0].to(device)
        audio = model(video)
        audio_path = os.path.join(output_dir, vid_fn.replace('.mp4', '.wav'))
        torchaudio.save(audio_path, audio.cpu(), model.audio_sample_rate)


def set_seed(seed: int = 42):
    """Set seed for torch."""
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    import fire
    fire.Fire()