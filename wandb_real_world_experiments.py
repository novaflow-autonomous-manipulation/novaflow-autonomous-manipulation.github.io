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

    # Check if flow_visualization_object exists in summary
    if not (run.summary and 'flow_visualization_object' in run.summary):
        print("flow_visualization_object not found in run summary")
        print(f"Available summary keys: {list(run.summary.keys()) if run.summary else 'None'}")
        return output_dir

    flow_viz = run.summary['flow_visualization_object']
    print(f"Found flow_visualization_object: {type(flow_viz)}")

    # Print available keys for debugging
    if hasattr(flow_viz, 'keys'):
        keys = list(flow_viz.keys())
        print(f"Available keys: {keys}")
        for key in keys[:5]:
            value = flow_viz[key]
            print(f"  {key}: {type(value)} - {str(value)[:100]}...")

    # Try to access image data directly
    import numpy as np

    image_data = None
    for key in ['image', '_image', 'data', 'value']:
        if key in flow_viz:
            print(f"Found '{key}' key")
            image_data = flow_viz[key]
            break

    if image_data is not None and isinstance(image_data, np.ndarray):
        print(f"Image data shape: {image_data.shape}")

        # Convert numpy array to PIL Image
        if len(image_data.shape) == 3 and image_data.shape[2] == 3:  # RGB image
            flow_pil_image = Image.fromarray((image_data * 255).astype(np.uint8))
        elif len(image_data.shape) == 2:  # Grayscale image
            flow_pil_image = Image.fromarray((image_data * 255).astype(np.uint8), mode='L')
        else:
            print(f"Unsupported image data shape: {image_data.shape}")
            return output_dir

        output_path = os.path.join(output_dir, f"flow_visualization_{run.name}.jpg")
        
        # Convert to RGB before saving as JPEG
        if flow_pil_image.mode != 'RGB':
            flow_pil_image = flow_pil_image.convert('RGB')
            
        flow_pil_image.save(output_path, 'JPEG', quality=95)
        print(f"Saved flow visualization to: {output_path}")

    # Handle image-file type objects
    if flow_viz.get('_type') == 'image-file' and 'path' in flow_viz:
        image_path = flow_viz['path']
        print(f"Image path: {image_path}")

        # Find and download the file
        files = run.files()
        target_file = None

        for file in files:
            if file.name == image_path:
                target_file = file
                break

        if not target_file:
            print(f"Target file {image_path} not found in run files")
            return output_dir

        print(f"Found target file: {target_file.name} ({target_file.size} bytes)")

        # Download the file
        target_file.download(replace=True)

        # Find the downloaded file - check both full path and basename
        import shutil
        expected_filename = os.path.basename(image_path)
        downloaded_file = None

        # First check if the full path exists (wandb preserves directory structure)
        if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
            downloaded_file = image_path
            print(f"Found downloaded file at full path: {downloaded_file}")
        # Then check if just the filename exists
        elif os.path.exists(expected_filename) and os.path.getsize(expected_filename) > 0:
            downloaded_file = expected_filename
            print(f"Found downloaded file at basename: {downloaded_file}")
        else:
            # List current directory to see what's actually there
            current_files = os.listdir('.')
            print(f"Files in current directory after download: {current_files[:10]}")  # Show first 10
            print("Downloaded file not found or empty")
            return output_dir

        # Directly copy the downloaded file to the desired output path
        output_path = os.path.join(output_dir, f"flow_visualization_{run.name}.jpg")
        
        # Open the image, convert to RGB, and save as JPEG
        try:
            with Image.open(downloaded_file) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(output_path, 'JPEG', quality=95)
            print(f"Converted and saved flow visualization to: {output_path}")
        except Exception as e:
            print(f"Error converting image to JPEG: {e}")
            # Fallback to simple copy if conversion fails
            import shutil
            shutil.copy2(downloaded_file, output_path)
            print(f"Copied flow visualization to: {output_path}")


        # Clean up the downloaded file
        if os.path.exists(downloaded_file):
            os.unlink(downloaded_file)

        # Read the copied image for concatenation
        try:
            flow_pil_image = Image.open(output_path)
        except Exception as e:
            print(f"Error reading flow visualization image: {e}")
            return output_dir

    else:
        print(f"Not a recognized wandb media type: {getattr(flow_viz, '_type', 'unknown')}")
        print(f"Available keys: {list(flow_viz.keys()) if hasattr(flow_viz, 'keys') else 'No keys'}")

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

    # First, check for and download real_experiement_camera_0 video if it exists
    if run.summary and 'real_experiement_camera_0' in run.summary:
        print("\nFound real_experiement_camera_0 video in run summary")
        video_obj = run.summary['real_experiement_camera_0']
        print(f"Video object type: {type(video_obj)}")

        if hasattr(video_obj, 'keys'):
            print(f"Video object keys: {list(video_obj.keys())}")

        # Handle video file downloads
        if video_obj.get('_type') == 'video-file' and 'path' in video_obj:
            video_path = video_obj['path']
            print(f"Video path: {video_path}")

            # Find and download the video file
            files = run.files()
            target_file = None

            for file in files:
                if file.name == video_path:
                    target_file = file
                    break

            if not target_file:
                print(f"Target video file {video_path} not found in run files")
            else:
                print(f"Found target video file: {target_file.name} ({target_file.size} bytes)")

                # Download the video file
                target_file.download(replace=True)

                # Find the downloaded file
                import shutil
                expected_filename = os.path.basename(video_path)
                downloaded_file = None

                # First check if the full path exists
                if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                    downloaded_file = video_path
                    print(f"Found downloaded video at full path: {downloaded_file}")
                # Then check if just the filename exists
                elif os.path.exists(expected_filename) and os.path.getsize(expected_filename) > 0:
                    downloaded_file = expected_filename
                    print(f"Found downloaded video at basename: {downloaded_file}")
                else:
                    # List current directory to see what's there
                    current_files = os.listdir('.')
                    print(f"Files in current directory after download: {current_files[:10]}")
                    print("Downloaded video file not found or empty")
                    downloaded_file = None

                if downloaded_file:
                    # Copy the video to the same output directory as the flow visualization
                    video_output_path = os.path.join(output_dir, f"real_experiement_camera_0_{run.name}.mp4")
                    shutil.copy2(downloaded_file, video_output_path)
                    print(f"Saved video to: {video_output_path}")

                    # Clean up the downloaded file
                    if os.path.exists(downloaded_file):
                        os.unlink(downloaded_file)

        elif hasattr(video_obj, '_type') and video_obj._type in ['video', 'mp4']:
            print("Direct video object found, attempting to save...")
            # Handle direct video objects if they exist
            # This would depend on the specific wandb video object structure
            print("Direct video object handling not implemented yet")
        else:
            print(f"Not a recognized video type: {getattr(video_obj, '_type', 'unknown')}")
    else:
        print("\nreal_experiement_camera_0 not found in run summary")
        if run.summary:
            print(f"Available summary keys: {list(run.summary.keys())}")

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
        download_and_visualize_flow_from_run(selected_run, "/Users/hongyu/Documents/RAI_project/Figure")