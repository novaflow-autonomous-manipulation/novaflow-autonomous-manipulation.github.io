
document.addEventListener('DOMContentLoaded', function() {
  const taskButtonsContainer = document.getElementById('task-buttons');
  const taskViewer = document.getElementById('task-viewer');
  const initialObservationImg = document.getElementById('initial-observation');
  const generatedVideo = document.getElementById('generated-video');
  const executionVideo = document.getElementById('execution-video');

//   const buildSrc = (task) => `assets/viser-client/?playbackPath=assets/videos/real-world/${task}.viser&initialCameraPosition=0,0,0&initialCameraLookAt=0.0,0.0,1.0&initialCameraUp=0.0,-1.0,0.0`;
    const buildSrc = (task) => `assets/viser-client/?playbackPath=../viser-client/${task}_recording.viser&initialCameraPosition=0,0,0&initialCameraLookAt=0.0,0.0,1.0&initialCameraUp=0.0,-1.0,0.0`;
  
  const updateMediaSources = (task) => {
    // a dictionary that maps tasks to their media files
    const media_path = `assets/viser-client/img/${task}`
    initialObservationImg.src = `${media_path}/start.png`;
    generatedVideo.src = `${media_path}/wan.mp4`;
    executionVideo.src = `${media_path}/camera0.mp4`;
    taskViewer.src = buildSrc(task);
  }

  taskButtonsContainer.addEventListener('click', function(e) {
    if (e.target.classList.contains('task-button')) {
      // Remove active class from all buttons
      taskButtonsContainer.querySelectorAll('.task-button').forEach(button => {
        button.classList.remove('active');
      });

      // Add active class to the clicked button
      e.target.classList.add('active');

      // Update the iframe source
      const selectedTask = e.target.dataset.value;
      updateMediaSources(selectedTask);
    }
  });

  // Set the initial view
  const initialTask = taskButtonsContainer.querySelector('.task-button.active').dataset.value;
  updateMediaSources(initialTask);
});
