// Homepage statistics functionality
document.addEventListener("DOMContentLoaded", function () {
  // GitHub username
  const username = "stibiums";

  // Initialize counters with animation
  function animateCounter(element, finalValue, duration = 1000) {
    const startValue = 0;
    const increment = finalValue / (duration / 16); // 60fps
    let currentValue = startValue;

    const timer = setInterval(() => {
      currentValue += increment;
      if (currentValue >= finalValue) {
        currentValue = finalValue;
        clearInterval(timer);
      }
      element.textContent = Math.floor(currentValue);
    }, 16);
  }

  // Fetch GitHub statistics
  async function fetchGitHubStats() {
    try {
      const response = await fetch(`https://api.github.com/users/${username}`);
      if (response.ok) {
        const data = await response.json();

        // Update GitHub repos count
        const reposElement = document.getElementById("github-repos");
        if (reposElement) {
          animateCounter(reposElement, data.public_repos || 0);
        }
      } else {
        console.warn("Failed to fetch GitHub user data");
        document.getElementById("github-repos").textContent = "~";
      }
    } catch (error) {
      console.warn("Error fetching GitHub stats:", error);
      document.getElementById("github-repos").textContent = "~";
    }
  }

  // Fetch GitHub stars count
  async function fetchGitHubStars() {
    try {
      const response = await fetch(`https://api.github.com/users/${username}/repos?per_page=100`);
      if (response.ok) {
        const repos = await response.json();
        const totalStars = repos.reduce((sum, repo) => sum + (repo.stargazers_count || 0), 0);

        const starsElement = document.getElementById("github-stars");
        if (starsElement) {
          animateCounter(starsElement, totalStars);
        }
      } else {
        document.getElementById("github-stars").textContent = "~";
      }
    } catch (error) {
      console.warn("Error fetching GitHub stars:", error);
      document.getElementById("github-stars").textContent = "~";
    }
  }

  // Count blog posts - try multiple sources
  async function countBlogPosts() {
    try {
      // Try to fetch posts from Jekyll's JSON feed if available
      const response = await fetch("/feed.json");
      if (response.ok) {
        const data = await response.json();
        const postsCount = data.items ? data.items.length : 0;

        const postsElement = document.getElementById("blog-posts");
        if (postsElement) {
          animateCounter(postsElement, postsCount);
        }
        return;
      }
    } catch (error) {
      console.log("JSON feed not available, using fallback");
    }

    // Try to fetch from sitemap or other sources
    try {
      const response = await fetch("/sitemap.xml");
      if (response.ok) {
        const text = await response.text();
        // Count URLs that match blog post pattern
        const postMatches = text.match(/\/\d{4}\/\d{2}\/\d{2}\//g);
        if (postMatches) {
          const postsElement = document.getElementById("blog-posts");
          if (postsElement) {
            animateCounter(postsElement, postMatches.length);
          }
          return;
        }
      }
    } catch (error) {
      console.log("Sitemap not available, using manual count");
    }

    // Fallback: manual count based on actual posts (updated to match your _posts folder)
    const postsElement = document.getElementById("blog-posts");
    if (postsElement) {
      animateCounter(postsElement, 11); // Current actual count: 11 posts
    }
  }

  // Count study notes (simulate notes counting)
  async function countStudyNotes() {
    try {
      // Try to count notes from site structure
      // This is a rough estimate based on typical academic notes
      const notesElement = document.getElementById("study-notes");
      if (notesElement) {
        // Estimate based on CS231n, algorithms, Rust, etc.
        animateCounter(notesElement, 15);
      }
    } catch (error) {
      const notesElement = document.getElementById("study-notes");
      if (notesElement) {
        animateCounter(notesElement, 15);
      }
    }
  }

  // Theme-aware GitHub stats images
  function updateGitHubStatsTheme() {
    const isDark = document.documentElement.getAttribute("data-theme") === "dark";
    const githubStatsImg = document.querySelector(".github-stats-img");
    const githubLangsImg = document.querySelector(".github-langs-img");

    if (githubStatsImg && githubLangsImg) {
      const baseStatsUrl = `https://github-readme-stats.vercel.app/api?username=${username}&show_icons=true&hide_border=true`;
      const baseLangsUrl = `https://github-readme-stats.vercel.app/api/top-langs/?username=${username}&layout=compact&hide_border=true`;

      if (isDark) {
        githubStatsImg.src = baseStatsUrl + "&theme=dark&title_color=2698ba&icon_color=2698ba&text_color=c9c9c9&bg_color=1c1c1d";
        githubLangsImg.src = baseLangsUrl + "&theme=dark&title_color=2698ba&text_color=c9c9c9&bg_color=1c1c1d";
      } else {
        githubStatsImg.src = baseStatsUrl + "&theme=transparent&title_color=b509ac&icon_color=b509ac&text_color=333";
        githubLangsImg.src = baseLangsUrl + "&theme=transparent&title_color=b509ac&text_color=333";
      }
    }
  }

  // Initialize all statistics
  function initializeStats() {
    fetchGitHubStats();
    fetchGitHubStars();
    countBlogPosts();
    countStudyNotes();
    updateGitHubStatsTheme();
  }

  // Observer for theme changes
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      if (mutation.type === "attributes" && mutation.attributeName === "data-theme") {
        updateGitHubStatsTheme();
      }
    });
  });

  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ["data-theme"],
  });

  // Start initialization
  initializeStats();

  // Refresh stats every 5 minutes
  setInterval(initializeStats, 300000);
});
