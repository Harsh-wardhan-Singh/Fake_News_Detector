// Initialize theme
const themeToggle = document.getElementById("theme-toggle");
const currentTheme = localStorage.getItem("theme") || "dark";
document.body.classList.add(currentTheme + "-theme");
themeToggle.textContent = currentTheme === "dark" ? "🌙" : "☀️";

// Toggle handler
themeToggle.onclick = () => {
  const newTheme = document.body.classList.contains("dark-theme") ? "light" : "dark";
  document.body.classList.replace(
    document.body.classList.contains("dark-theme") ? "dark-theme" : "light-theme",
    newTheme + "-theme"
  );
  themeToggle.textContent = newTheme === "dark" ? "🌙" : "☀️";
  localStorage.setItem("theme", newTheme);
};
