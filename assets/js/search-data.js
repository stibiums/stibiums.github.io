// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-projects",
          title: "projects",
          description: "A growing collection of your cool projects.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-闪击春城",
        
          title: "闪击春城",
        
        description: "说走就走，不留遗憾",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/07/21/Kunming/";
          
        },
      },{id: "post-notes-of-rustlearning",
        
          title: "notes of Rustlearning",
        
        description: "这是我学习Rust语言的笔记.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/07/08/noteofRust/";
          
        },
      },{id: "post-大连",
        
          title: "大连",
        
        description: "开心就好",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/07/06/dalian/";
          
        },
      },{id: "post-notes-of-algorithm-and-data",
        
          title: "notes of algorithm and data",
        
        description: "这是关于算法与数据结构的学习笔记.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/07/01/notesofalgorithm_and_data/";
          
        },
      },{id: "post-notes-of-cv",
        
          title: "notes of CV",
        
        description: "这是关于计算机视觉(CV)的学习笔记.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/07/01/notesofCS231n/";
          
        },
      },{id: "post-wordhub-单词学习软件",
        
          title: "wordhub-单词学习软件",
        
        description: "wordhub是一款专注于单词学习的应用程序.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/06/30/Wordhub/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-a-simple-inline-announcement",
          title: 'A simple inline announcement.',
          description: "",
          section: "News",},{id: "news-a-long-announcement-with-details",
          title: 'A long announcement with details',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/news/announcement_2/";
            },},{id: "news-a-simple-inline-announcement-with-markdown-emoji-sparkles-smile",
          title: 'A simple inline announcement with Markdown emoji! :sparkles: :smile:',
          description: "",
          section: "News",},{id: "notes-cs231n-1-numpy",
          title: 'CS231n - 1: Numpy',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/cs231n-ch01/";
            },},{id: "notes-cs231n-2-图像分类",
          title: 'CS231n - 2: 图像分类',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/cs231n-ch02/";
            },},{id: "projects-wordhub",
          title: 'WordHub',
          description: "高效简洁的单词学习软件",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%31%33%36%30%30%32%33%39%39%34@%71%71.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/stibiums", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
