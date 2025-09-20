import os
import wandb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tempfile
import cv2
import numpy as np
from PIL import Image
import shutil

def select_frames_from_video(video_path, n_frames=5):
    """
    Select N frames evenly spaced from a video file.
    The first and last frames of the video are always included.

    Args:
        video_path (str): Path to the video file
        n_frames (int): Number of frames to select (default: 5)

    Returns:
        list: List of PIL Image objects representing the selected frames
    """
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return []

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return []

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video: {total_frames} frames, {fps} FPS")

    if total_frames < n_frames:
        print(f"Video has fewer frames ({total_frames}) than requested ({n_frames}). Using all frames.")
        n_frames = total_frames

    # Calculate frame indices to select evenly spaced frames
    frame_indices = []
    if n_frames > 0:
        if n_frames == 1:
            # If only one frame is requested, take the middle one
            frame_indices.append(total_frames // 2)
        else:
            # Ensure the first and last frames are included
            frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
            frame_indices = sorted(list(set(frame_indices)))  # Ensure uniqueness and order

    print(f"Selecting frames at indices: {frame_indices}")

    # Extract the frames
    frames = []
    for frame_idx in frame_indices:
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # Read the frame
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB and then to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
            print(f"Extracted frame {frame_idx}")
        else:
            print(f"Failed to read frame {frame_idx}")

    cap.release()
    print(f"Successfully extracted {len(frames)} frames")
    return frames


def concatenate_images_horizontally(images, output_path=None):
    """
    Concatenate a list of PIL Images horizontally.

    Args:
        images (list): List of PIL Image objects
        output_path (str): Optional path to save the concatenated image

    Returns:
        PIL.Image: The concatenated image
    """
    if not images:
        print("No images to concatenate")
        return None

    # skip the second image
    images.pop(1)

    # Get the maximum height to align all images
    max_height = max(img.height for img in images)

    # Resize images to have the same height while maintaining aspect ratio
    resized_images = []
    total_width = 0

    for img in images:
        # Calculate new width to maintain aspect ratio
        aspect_ratio = img.width / img.height
        new_width = int(max_height * aspect_ratio)
        total_width += new_width

        # Resize the image
        resized_img = img.resize((new_width, max_height), Image.Resampling.LANCZOS)
        resized_images.append(resized_img)

    # Create a new image with the total width
    concatenated = Image.new('RGB', (total_width, max_height))

    # Paste each image
    x_offset = 0
    for img in resized_images:
        concatenated.paste(img, (x_offset, 0))
        x_offset += img.width

    if output_path:
        # Convert to RGB if it has an alpha channel (RGBA)
        if concatenated.mode == 'RGBA':
            concatenated = concatenated.convert('RGB')
        concatenated.save(output_path, 'JPEG', quality=95)
        print(f"Saved concatenated image to: {output_path}")

        # Copy the image to the real_world_experiments directory
        real_world_experiments_dir = os.path.join(os.path.dirname(os.path.dirname(output_path)), "../../../", "real_world_experiments")
        os.makedirs(real_world_experiments_dir, exist_ok=True)
        shutil.copy(output_path, real_world_experiments_dir)
        print(f"Copied concatenated image to: {real_world_experiments_dir}")

    return concatenated


def download_media_from_run(run, media_key, output_dir, output_filename=None):
    """
    Downloads a media file (image or video) from a wandb run summary.
    """
    if not (run.summary and media_key in run.summary):
        print(f"\n'{media_key}' not found in run summary.")
        if run.summary:
            print(f"Available summary keys: {list(run.summary.keys())}")
        return None

    media_obj = run.summary[media_key]
    print(f"\nFound '{media_key}' in run summary (type: {type(media_obj)})")

    if hasattr(media_obj, 'keys'):
        print(f"Media object keys: {list(media_obj.keys())}")

    media_type = media_obj.get('_type')
    if media_type not in ['video-file', 'image-file']:
        print(f"Not a recognized media type: {media_type}")
        return None

    if 'path' not in media_obj:
        print(f"'path' not found in media object for key '{media_key}'")
        return None

    media_path = media_obj['path']
    print(f"Media path from wandb: {media_path}")

    # Find and download the file
    files = run.files()
    target_file = None
    for file in files:
        if file.name == media_path:
            target_file = file
            break

    if not target_file:
        print(f"Target file '{media_path}' not found in run files.")
        return None

    print(f"Found target file: {target_file.name} ({target_file.size} bytes)")

    # Download the file to a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        target_file.download(root=tmpdir, replace=True)
        downloaded_file_path = os.path.join(tmpdir, target_file.name)

        if os.path.exists(downloaded_file_path) and os.path.getsize(downloaded_file_path) > 0:
            print(f"Successfully downloaded to: {downloaded_file_path}")

            if not output_filename:
                # keep original extension
                _, ext = os.path.splitext(os.path.basename(media_path))
                output_filename = f"{media_key}_{run.name}{ext}"

            final_output_path = os.path.join(output_dir, output_filename)
            shutil.copy2(downloaded_file_path, final_output_path)
            print(f"Saved media to: {final_output_path}")
            return final_output_path
        else:
            print("Downloaded file not found or is empty.")
            return None


def save_text_from_run(run, text_key, output_dir, output_filename=None):
    """
    Saves a text string from a wandb run summary to a text file.
    """
    if not (run.summary and text_key in run.summary):
        print(f"\n'{text_key}' not found in run summary.")
        if run.summary:
            print(f"Available summary keys: {list(run.summary.keys())}")
        return None

    text_content = run.summary[text_key]
    print(f"\nFound '{text_key}' in run summary.")

    if not isinstance(text_content, str):
        print(f"Content of '{text_key}' is not a string (type: {type(text_content)}).")
        return None

    if not output_filename:
        output_filename = f"{text_key}_{run.name}.txt"

    final_output_path = os.path.join(output_dir, output_filename)

    try:
        with open(final_output_path, 'w') as f:
            f.write(text_content)
        print(f"Saved text to: {final_output_path}")
        return final_output_path
    except Exception as e:
        print(f"Error saving text file: {e}")
        return None


def speed_up_video(video_path, speed_factor=2):
    """
    Speed up a video using ffmpeg.

    Args:
        video_path (str): Path to the input video file
        speed_factor (int): Factor by which to speed up the video (e.g., 2 for 2x)
    """
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return

    # Get the directory and filename
    directory, filename = os.path.split(video_path)
    name, ext = os.path.splitext(filename)

    # Define the output path for the sped-up video
    output_path = os.path.join(directory, f"{name}_{speed_factor}x{ext}")

    # Construct the ffmpeg command
    # Use -y to overwrite output file if it exists
    # Use -an to remove audio
    command = [
        "ffmpeg",
        "-i", video_path,
        "-y",
        "-filter:v", f"setpts={1/speed_factor}*PTS",
        "-an",  # No audio
        output_path
    ]

    print(f"Running ffmpeg command: {' '.join(command)}")

    try:
        # Run the command
        import subprocess
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("FFmpeg output:", result.stdout)
        print("FFmpeg error:", result.stderr)
        print(f"Successfully created sped-up video: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg: {e}")
        print("FFmpeg output:", e.stdout)
        print("FFmpeg error:", e.stderr)
    except FileNotFoundError:
        print("ffmpeg not found. Please ensure it is installed and in your system's PATH.")


def _matches_filters(run, filters):
    """Check if a run matches all the provided filters."""
    for filter_dict in filters:
        attribute = filter_dict.get('attribute', '')
        operator = filter_dict.get('operator', '==')
        value = filter_dict.get('value')
        attr_type = filter_dict.get('type', 'auto')  # auto, config, summary, run, tags

        # Get the actual value from the run
        actual_value = _get_attribute_value(run, attribute, attr_type)

        # Apply the comparison
        if not _compare_values(actual_value, operator, value):
            return False

    return True


def _get_attribute_value(run, attribute, attr_type='auto'):
    """Get the value of an attribute from a run."""
    if attr_type == 'auto':
        # Try to determine type from attribute name
        if attribute.startswith('config.'):
            attr_type = 'config'
            attribute = attribute[7:]  # Remove 'config.' prefix
        elif attribute.startswith('summary.'):
            attr_type = 'summary'
            attribute = attribute[8:]  # Remove 'summary.' prefix
        elif attribute in ['name', 'state', 'id', 'created_at']:
            attr_type = 'run'
        else:
            attr_type = 'run'  # Default to run attributes

    if attr_type == 'config':
        return run.config.get(attribute) if run.config else None
    elif attr_type == 'summary':
        return run.summary.get(attribute) if run.summary else None
    elif attr_type == 'tags':
        return run.tags if hasattr(run, 'tags') else []
    elif attr_type == 'run':
        if attribute == 'name':
            return run.name
        elif attribute == 'state':
            return run.state
        elif attribute == 'id':
            return run.id
        elif attribute == 'created_at':
            return run.created_at
        elif attribute == 'user':
            return run.user.username if run.user else None
        else:
            return getattr(run, attribute, None)

    return None


def _compare_values(actual_value, operator, expected_value):
    """Compare two values using the specified operator."""
    if operator == '==':
        return actual_value == expected_value
    elif operator == '!=':
        return actual_value != expected_value
    elif operator == '<':
        return actual_value < expected_value if actual_value is not None and expected_value is not None else False
    elif operator == '<=':
        return actual_value <= expected_value if actual_value is not None and expected_value is not None else False
    elif operator == '>':
        return actual_value > expected_value if actual_value is not None and expected_value is not None else False
    elif operator == '>=':
        return actual_value >= expected_value if actual_value is not None and expected_value is not None else False
    elif operator == 'contains':
        if isinstance(actual_value, str) and isinstance(expected_value, str):
            return expected_value in actual_value
        elif isinstance(actual_value, list):
            return expected_value in actual_value
        return False
    elif operator == 'in':
        if isinstance(expected_value, list):
            return actual_value in expected_value
        return False
    elif operator == 'startswith':
        if isinstance(actual_value, str) and isinstance(expected_value, str):
            return actual_value.startswith(expected_value)
        return False
    elif operator == 'endswith':
        if isinstance(actual_value, str) and isinstance(expected_value, str):
            return actual_value.endswith(expected_value)
        return False
    else:
        return False

def get_output_directory(run, base_output_dir="/Users/hongyu/Documents/RAI_project/Figure"):
    """
    Get the output directory path for a run.
    Creates the directory if it doesn't exist.

    Args:
        run: W&B run object
        base_output_dir: Base directory for outputs

    Returns:
        str: Path to the output directory
    """
    output_dir = os.path.join(base_output_dir, "media", "real_experiment", run.group, run.id)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_flow_visualization(run, base_output_dir="/Users/hongyu/Documents/RAI_project/Figure"):
    """
    Helper function to save flow visualization from a wandb run.
    Returns the output directory where files are saved.
    """
    print(f"Processing run: {run.name} (ID: {run.id})")

    # Create output directory based on run ID
    output_dir = get_output_directory(run, base_output_dir)
    print(f"Saving images to: {output_dir}")
    
    # Check if final concatenated image already exists
    concat_output_path = os.path.join(output_dir, f"flow_and_frames_{run.name}.jpg")
    if os.path.exists(concat_output_path):
        print(f"Concatenated image already exists: {concat_output_path}. Skipping.")
        return output_dir

    # Download flow visualization object
    flow_output_path = download_media_from_run(run, 'flow_visualization_object', output_dir, f"flow_visualization_{run.name}.jpg")

    if not flow_output_path:
        print("Failed to download flow visualization.")
        return output_dir
    
    try:
        flow_pil_image = Image.open(flow_output_path)
    except Exception as e:
        print(f"Error reading flow visualization image: {e}")
        return output_dir


    # Check if we have a flow visualization image (either from numpy array or file)
    if 'flow_pil_image' in locals():
        print("\nProcessing video frames for concatenation...")

        # Look for the corresponding video file
        video_filename = f"real_experiement_camera_0_{run.name}.mp4"
        video_path = os.path.join(output_dir, video_filename)

        if os.path.exists(video_path):
            print(f"Found video file: {video_path}")

            # Extract N frames from the video (default N=5)
            video_frames = select_frames_from_video(video_path, n_frames=4)

            if video_frames:
                print(f"Loaded flow visualization image: {flow_pil_image.size}")

                # Concatenate flow image with video frames horizontally
                images_to_concat = [flow_pil_image] + video_frames
                concat_output_path = os.path.join(output_dir, f"flow_and_frames_{run.name}.jpg")

                concatenated_image = concatenate_images_horizontally(images_to_concat, concat_output_path)

                if concatenated_image:
                    print(f"Successfully created concatenated image: {concatenated_image.size}")
                    print(f"Saved to: {concat_output_path}")
                else:
                    print("Failed to create concatenated image")

            else:
                print("No frames extracted from video")
        else:
            print(f"Video file not found: {video_path}")
    else:
        print("No flow visualization image available for concatenation")

    return output_dir


def visualize_flow_from_first_run(entity="bdaii", project="gizmo", filters=None):
    """
    Visualize flow_visualization_object from the first run matching the filters.
    """

    api = wandb.Api()

    # Get runs and apply filters
    runs = api.runs(f"{entity}/{project}")

    if filters:
        filtered_runs = []
        for run in runs:
            if _matches_filters(run, filters):
                filtered_runs.append(run)
        runs = filtered_runs

    if not runs:
        print("No runs found matching the filters.")
        return

    # Get the first run
    first_run = runs[0]
    print(f"Using first run: {first_run.name} (ID: {first_run.id})")

    # Use the shared function to save flow visualization
    output_dir = save_flow_visualization(first_run)


def download_and_visualize_flow_from_run(run, base_output_dir="/Users/hongyu/Documents/RAI_project/Figure"):
    """
    Download and visualize flow_visualization_object from a specific wandb run.
    """
    # Define output directory and check if final files already exist
    output_dir = get_output_directory(run, base_output_dir)
    video_output_path = os.path.join(output_dir, f"real_experiement_camera_0_{run.name}.mp4")
    flow_output_path = os.path.join(output_dir, f"flow_visualization_{run.name}.jpg")
    concat_output_path = os.path.join(output_dir, f"flow_and_frames_{run.name}.jpg")

    if os.path.exists(video_output_path) and os.path.exists(flow_output_path) and os.path.exists(concat_output_path):
        print(f"All files already exist for run {run.name} (ID: {run.id}). Skipping.")
        return

    # Media keys to download
    media_keys_to_download = ['real_experiement_camera_0', 'first_rgb', 'wan_video']
    downloaded_paths = {}

    for key in media_keys_to_download:
        path = download_media_from_run(run, key, output_dir)
        if path:
            downloaded_paths[key] = path

    # Save text prompts
    save_text_from_run(run, 'wan_prompt_extended', output_dir)
    save_text_from_run(run, 'wan_prompt_original', output_dir)

    # Speed up videos if they were downloaded
    if 'real_experiement_camera_0' in downloaded_paths:
        speed_up_video(downloaded_paths['real_experiement_camera_0'], speed_factor=2)
    
    if 'wan_video' in downloaded_paths:
        speed_up_video(downloaded_paths['wan_video'], speed_factor=2)

    # Now process the flow visualization
    save_flow_visualization(run, base_output_dir)


if __name__ == "__main__":
    # Filter runs that have "drawer_open" in their name
    base_filters = [
        {"attribute": "summary.real_experiment_result", "operator": "contains", "value": "success"},
    ]
    id_filters = [
        {"attribute": "id", "operator": "contains", "value": "hjiux8ft"},
        {"attribute": "id", "operator": "contains", "value": "phf8ho3j"},
        {"attribute": "id", "operator": "contains", "value": "wmmoiuwt"},
        {"attribute": "id", "operator": "contains", "value": "b1tsdqyy"},
        {"attribute": "id", "operator": "contains", "value": "zjq8z2q9"},
        {"attribute": "id", "operator": "contains", "value": "mfgdbnuu"},
        {"attribute": "id", "operator": "contains", "value": "4w92zcly"},
    ]

    # Get the runs
    print("Getting filtered runs...")
    api = wandb.Api()
    runs = api.runs(f"bdaii/gizmo")
    filtered_runs = []
    for run in runs:
        if _matches_filters(run, base_filters):
            id_match = False
            for id_filter in id_filters:
                if _matches_filters(run, [id_filter]):
                    id_match = True
                    break
            if id_match:
                filtered_runs.append(run)

    if not filtered_runs:
        print("No runs found matching the filters.")
        exit(1)

    print(f"Found {len(filtered_runs)} runs matching filters.")
    for i in range(len(filtered_runs)):
        selected_run = filtered_runs[i]
        print(f"Using run: {selected_run.name} (ID: {selected_run.id} Group: {selected_run.group})")
        download_and_visualize_flow_from_run(selected_run, "/home/holi/Desktop/novaflow-autonomous-manipulation.github.io/assets/wandb")