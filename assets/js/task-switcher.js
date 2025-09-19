
document.addEventListener('DOMContentLoaded', function() {
  const taskButtonsContainer = document.getElementById('task-buttons');
  const taskViewer = document.getElementById('task-viewer');

//   const buildSrc = (task) => `assets/viser-client/?playbackPath=assets/videos/real-world/${task}.viser&initialCameraPosition=0,0,0&initialCameraLookAt=0.0,0.0,1.0&initialCameraUp=0.0,-1.0,0.0`;
    const buildSrc = (task) => `assets/viser-client/?playbackPath=../recording.viser&initialCameraPosition=0,0,0&initialCameraLookAt=0.0,0.0,1.0&initialCameraUp=0.0,-1.0,0.0`;
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
      taskViewer.src = buildSrc(selectedTask);
    }
  });

  // Set the initial view
  const initialTask = taskButtonsContainer.querySelector('.task-button.active').dataset.value;
  taskViewer.src = buildSrc(initialTask);
});
