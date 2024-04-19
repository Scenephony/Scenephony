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

    chord = Chord(ch_str)
    midi_num = 0
    for note in chord.components_with_pitch(root_pitch=4):
        midi_num += note_name_to_number(note)
    return midi_num / len(chord.components())