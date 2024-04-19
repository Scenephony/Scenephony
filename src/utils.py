import torch
from pretty_midi import note_name_to_number
from pychord import Chord


def chord_to_number(ch_str: str) -> float:
    """Converts a chord string to a number.
    
    Args:
        ch_str: Input chord string. Example: "C", "Cm", "Cm7", "Co"/"Cdim".
    """
    # Preprocess chord string
    ch_str = ch_str.replace("o", "dim")
    ch_str = ch_str.replace("+", "aug")
    if ch_str.endswith("M"):
        ch_str = ch_str[:-1]

    chord = Chord(ch_str)
    midi_num = 0
    for note in chord.components_with_pitch(root_pitch=4):
        midi_num += note_name_to_number(note)
    return midi_num / len(chord.components())


def pad_chords(chords, pad_value=0):
    # Determine the maximum length of any chord sequence in the list
    max_len = max(tensor.size(0) for tensor in chords)

    # Pad each chord tensor to the maximum length found
    padded_chords = [torch.nn.functional.pad(tensor, (0, max_len - tensor.size(0)), value=pad_value) for tensor in
                     chords]

    # Stack the padded tensors into a single tensor
    return torch.stack(padded_chords)

