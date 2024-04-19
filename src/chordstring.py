def generate_chords():

    # All notes
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    # All chords
    chord_types = [
        "", "m", "dim", "aug", "M7", "m7", "7", "mM7", "m7♭5", "dim7", "aug7", "augM7",
        "sus2", "sus4", "6", "m6", "9", "M9", "m9", "11", "M11", "m11", "13", "M13", "m13",
        "add9", "add11", "add13"
    ]

    all_chords = [note + chord_type for note in notes for chord_type in chord_types]

    return all_chords

def MSE(predict, target, chord_dict):          # predict is model_output_list_of_chord, target is target_list_of_chords

    squared_diffs = []
    for pred, tar in zip(predict, target):
        if pred in chord_dict and tar in chord_dict:
            pred_index = chord_dict[pred]
            tar_index = chord_dict[tar]
            squared_diffs.append((pred_index - tar_index) ** 2)
        else:
            print(f"Chord '{pred}' or '{tar}' is not in the sorted chord list.")
    
    mse = sum(squared_diffs) / len(squared_diffs)
    return mse

if __name__ == "__main__":
    chords = generate_chords()
    chord_dict = {chord : idx for idx, chord in enumerate(chords)} # Create dictionary for faster access

    predicted_chords = ["C", "Am7", "F7", "G7", "Em", "Dm7", "G", "C7", "Fmaj7", "Bdim", "E", "A7"]
    target_chords = ["C", "Am", "F", "G", "Em7", "Dm", "G7", "Cmaj7", "F7", "Bdim", "E7", "Am7"]
    error = MSE(predicted_chords, target_chords, chord_dict)
     
    print(error)

