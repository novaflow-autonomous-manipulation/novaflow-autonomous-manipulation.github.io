// Hero video initialization
function startHero() {
  const vid = document.getElementById('bg-video');
  if (!vid) return;
  vid.play().catch(() => {});
  vid.style.opacity = 1;
}