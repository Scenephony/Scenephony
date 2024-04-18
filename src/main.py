import os
import torch
import torch.utils
import torchaudio
from math import ceil
from torchvision.io import read_video
from tqdm import tqdm
from typing import Optional

from model import ScenephonyModel, ScenephonyModelConfig


def preproc_vid(video: torch.Tensor, sample_frames: int, new_H: int, new_W: int) -> torch.Tensor:
    """Sample frames from the video, one every ceil(T / sample_frames) frames. In addition, resize
    each frame to have frame size new_h x new_w.
    
    Args:
        video: Input video frames of shape (T, C, H, W).
        sample_frames: Number of frames to sample.
        new_H: New height of the frame.
        new_W: New width of the frame.
    
    Returns:
        Sampled video frames of shape (sample_frames, C, new_H, new_W).
    """
    T = video.size(0)
    step = ceil(T / sample_frames)
    res = torch.nn.functional.interpolate(video[::step], (new_H, new_W))
    
    assert res.size(0) == sample_frames
    return res


def train(
    data_dir: str,
    output_dir: str,
    seed: int = 42,
    batch_size: int = 4,
    num_epochs: int = 10,
    lr: float = 1e-3,
    log_freq: int = 100,
    checkpoint_path: Optional[str] = None
):
    """Train the Scenephony model.

    Args:
        data_dir: Path to the training dataset.
        output_dir: Path to save the trained model.
        seed: Random seed.
        batch_size: Batch size.
        num_epochs: Number of epochs.
        lr: Learning rate.
        log_freq: Logging frequency.
        checkpoint_path: Path to optionally load a checkpoint before starting training.
    """
    # Initial setup
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup model, optimizer and loss function
    config = ScenephonyModelConfig()
    model = ScenephonyModel(config)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Load videos and target audios from data_dir files, then create dataloader
    videos, audios = [], []
    for fn in os.listdir(data_dir):
        fn_no_ext = fn.split('.')[0]
        video_path = os.path.join(data_dir, fn_no_ext + ".mp4")
        # audio_path = os.path.join(data_dir, fn_no_ext + ".wav")
        video = read_video(
            video_path, 
            start_pts=10, end_pts=20, pts_unit="sec", 
            output_format="TCHW"
        )[0].to(device)

        # Resize video to have frame size of 224x224
        video = preproc_vid(
            video, config.sample_frames, config.hidden_frame_h, config.hidden_frame_w
        )
        videos.append(video)

    dataset = torch.utils.data.TensorDataset(torch.stack(videos), torch.stack(audios))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Start training
    for epoch in range(num_epochs):
        for i, (video, audio) in tqdm(enumerate(dataloader)):
            model.train()
            optimizer.zero_grad()
            pred_audio = model(video)
            loss = criterion(pred_audio, audio)
            loss.backward()
            optimizer.step()

            if i % log_freq == 0:
                print(f"Epoch {epoch}, Iter {i}, Loss: {loss.item()}")

        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_{epoch}.pth"))

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
    config = ScenephonyModelConfig()
    model = ScenephonyModel(config)
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    # Load data, inference and save audio files
    for vid_fn in tqdm(os.listdir(data_dir)):
        vid_path = os.path.join(data_dir, vid_fn)
        video = read_video(
            vid_path, 
            start_pts=10, end_pts=20, pts_unit="sec", 
            output_format="TCHW"
        )[0].to(device)    # shape: (T, C, H, W)
        video = preproc_vid(
            video, config.sample_frames, config.hidden_frame_h, config.hidden_frame_w
        )
        print(video.shape)

        audio = model(video.unsqueeze(0))   # inp_shape: (1, T, C, H, W)
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