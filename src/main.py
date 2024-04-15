import os
import torch
import torch.utils
import torchaudio
from torchvision.io import read_video
from tqdm import tqdm
from typing import Optional

from model import ScenephonyModel


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
    model = ScenephonyModel()
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    # Load videos and target audios from data_dir files, then create dataloader
    videos, audios = [], []
    for fn in os.listdir(data_dir):
        # Process video and audio at the same iter, skip audios iters
        if fn.endswith('.wav'):
            continue

        fn_no_ext = fn.split('.')[0]
        video_path = os.path.join(data_dir, fn_no_ext + ".mp4")
        audio_path = os.path.join(data_dir, fn_no_ext + ".wav")
        videos.append(read_video(video_path, output_format="TCHW")[0].to(device))
        audios.append(torchaudio.load(audio_path)[0].to(device))

    # TODO: Handle videos with different num frames (T) and audio length
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
    model = ScenephonyModel()
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    # Load data, inference and save audio files
    for vid_fn in tqdm(os.listdir(data_dir)):
        vid_path = os.path.join(data_dir, vid_fn)
        video = read_video(vid_path, output_format="TCHW")[0].to(device)    # shape: (T, C, H, W)
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