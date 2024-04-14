import cv2
import os

def extract_key_frames(video_path, output_folder, num_frames=15):
    # Check if output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Capture the video from the file specified
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the interval between frames to be extracted
    interval = total_frames // num_frames

    # Frame index initialization
    current_frame = 0

    # Read through the video file
    while True:
        ret, frame = cap.read()
        # If frame reading was not successful, break the loop
        if not ret:
            break
        
        # If the current frame number is in the list of frames to extract
        if current_frame % interval == 0:
            # Save the frame
            cv2.imwrite(os.path.join(output_folder, f"frame_{current_frame}.png"), frame)
            # If we have reached the required number of frames, break the loop
            if len(os.listdir(output_folder)) == num_frames:
                break
        
        current_frame += 1

    # Release the video capture object and close all frames
    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_file_path = "input.mp4"  # Path to the video file
output_directory = "extracted_frames"  # Directory where frames will be saved
extract_key_frames(video_file_path, output_directory)

