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

  // Count projects
  async function countProjects() {
    try {
      // Try to fetch from projects page to count projects
      const response = await fetch("/projects/");
      if (response.ok) {
        const text = await response.text();
        // Count project cards or links in the projects page
        const projectMatches = text.match(/class="[^"]*project[^"]*"[^>]*>/gi) || text.match(/href="[^"]*\/projects\/[^"]*"/g);
        if (projectMatches) {
          const projectsElement = document.getElementById("projects-count");
          if (projectsElement) {
            // Filter unique projects to avoid duplicates
            const uniqueProjects = new Set(projectMatches);
            animateCounter(projectsElement, uniqueProjects.size);
          }
          return;
        }
      }
    } catch (error) {
      console.log("Projects page not available, using manual count");
    }

    // Try to count from sitemap
    try {
      const response = await fetch("/sitemap.xml");
      if (response.ok) {
        const text = await response.text();
        // Count URLs that match project pattern
        const projectMatches = text.match(/\/projects\/[^<]+/g);
        if (projectMatches) {
          const projectsElement = document.getElementById("projects-count");
          if (projectsElement) {
            animateCounter(projectsElement, projectMatches.length);
          }
          return;
        }
      }
    } catch (error) {
      console.log("Sitemap not available for projects count");
    }

    // Fallback: manual count based on actual projects (updated to match your files)
    const projectsElement = document.getElementById("projects-count");
    if (projectsElement) {
      // Current count: 1 project (WordHub) + GitHub projects
      // You can add GitHub projects count here or keep it simple
      animateCounter(projectsElement, 1 + 3); // 1 showcase project + estimate of 3 GitHub projects
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
    countProjects();
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

  // Add click event handlers for clickable elements
  function initializeClickHandlers() {
    // Handle clicks on stat cards and skill tags
    document.addEventListener("click", function (event) {
      const clickableElement = event.target.closest(".clickable");
      if (clickableElement && clickableElement.dataset.url) {
        const url = clickableElement.dataset.url;

        // Check if it's an external link
        if (url.startsWith("http://") || url.startsWith("https://")) {
          window.open(url, "_blank", "noopener,noreferrer");
        } else {
          // Internal navigation
          window.location.href = url;
        }
      }
    });

    // Add keyboard accessibility (Enter key)
    document.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        const clickableElement = event.target.closest(".clickable");
        if (clickableElement && clickableElement.dataset.url) {
          event.preventDefault();
          const url = clickableElement.dataset.url;

          if (url.startsWith("http://") || url.startsWith("https://")) {
            window.open(url, "_blank", "noopener,noreferrer");
          } else {
            window.location.href = url;
          }
        }
      }
    });
  }

  // Initialize click handlers
  initializeClickHandlers();

  // Start initialization
  initializeStats();

  // Refresh stats every 5 minutes
  setInterval(initializeStats, 300000);
});
