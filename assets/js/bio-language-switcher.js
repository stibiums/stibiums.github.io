// Simple language switcher for personal bio only
document.addEventListener("DOMContentLoaded", function () {
  let currentLanguage = localStorage.getItem("bio-language") || "zh";

  // Initialize language on page load
  switchBioLanguage(currentLanguage);

  // Language switcher button event
  const bioLangBtn = document.getElementById("bio-language-btn");
  if (bioLangBtn) {
    bioLangBtn.addEventListener("click", function () {
      currentLanguage = currentLanguage === "zh" ? "en" : "zh";
      switchBioLanguage(currentLanguage);
      localStorage.setItem("bio-language", currentLanguage);
    });
  }

  function switchBioLanguage(lang) {
    // Hide all bio language content
    document.querySelectorAll(".bio-zh, .bio-en").forEach(function (element) {
      element.style.display = "none";
    });

    // Show selected language bio content
    document.querySelectorAll(".bio-" + lang).forEach(function (element) {
      element.style.display = "block";
    });

    // Update button text
    const bioLangBtn = document.getElementById("bio-language-btn");
    if (bioLangBtn) {
      bioLangBtn.textContent = lang === "zh" ? "EN" : "中文";
      bioLangBtn.title = lang === "zh" ? "Switch to English" : "切换到中文";
    }
  }
});
