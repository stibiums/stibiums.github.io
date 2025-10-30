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
        },{id: "nav-notes",
          title: "notes",
          description: "学习笔记与技术分享 / Study Notes &amp; Technical Sharing",
          section: "Navigation",
          handler: () => {
            window.location.href = "/notes/";
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
        },{id: "post-notes-of-ml",
        
          title: "notes of ML",
        
        description: "2025 机器学习旁听",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/09/08/notesof_ML/";
          
        },
      },{id: "post-notes-of-vci",
        
          title: "notes of VCI",
        
        description: "这是关于可视计算与交互概论(VCI)的学习笔记.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/09/08/notesofvci/";
          
        },
      },{id: "post-notes-of-aip",
        
          title: "notes of AIP",
        
        description: "这是关于人工智能编程(AIP)的学习笔记.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/09/08/notesofaip/";
          
        },
      },{id: "post-notes-of-ai-math-fundamentals",
        
          title: "notes of AI Math Fundamentals",
        
        description: "AI数学基础课程笔记 - 概率论部分",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/09/08/notesof_aimath/";
          
        },
      },{id: "post-notes-of-ics",
        
          title: "notes of ICS",
        
        description: "这是关于计算机系统导论(ICS)的学习笔记.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/09/08/notesof_ICS/";
          
        },
      },{id: "post-再游春城",
        
          title: "再游春城",
        
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
          
            window.location.href = "/blog/2025/07/01/notesofCV/";
          
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
            },},{id: "notes-cs231n-3-神经网络到cnn",
          title: 'CS231n - 3: 神经网络到CNN',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/cs231n-ch03/";
            },},{id: "notes-vci-1-颜色-颜色感知与可视化",
          title: 'VCI - 1: 颜色，颜色感知与可视化',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/vci-ch01/";
            },},{id: "notes-ics-第二讲-位-字节和整数-bits-bytes-and-integers",
          title: 'ICS - 第二讲：位、字节和整数 (Bits, Bytes, and Integers)',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/ICS-ch01/";
            },},{id: "notes-ml-1-linear-regression-线性回归",
          title: 'ML - 1: Linear Regression (线性回归)',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/ML-ch01/";
            },},{id: "notes-数据结构与算法-第1章-概论",
          title: '数据结构与算法 - 第1章: 概论',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/dsa-ch01/";
            },},{id: "notes-人工智能中的编程-第2章-并行编程-parallel-programming",
          title: '人工智能中的编程 - 第2章: 并行编程（Parallel Programming）',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/AIP-ch02/";
            },},{id: "notes-数据结构与算法-第2章-线性表",
          title: '数据结构与算法 - 第2章: 线性表',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/dsa-ch02/";
            },},{id: "notes-ai数学基础-第1讲-概率论基础概念",
          title: 'AI数学基础 - 第1讲: 概率论基础概念',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/aimath-ch01/";
            },},{id: "notes-ai数学基础-第2讲-条件概率与独立性",
          title: 'AI数学基础 - 第2讲: 条件概率与独立性',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/aimath-ch02/";
            },},{id: "notes-vci-2-显示",
          title: 'VCI - 2: 显示',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/vci-ch02/";
            },},{id: "notes-vci-3-2d图形绘制",
          title: 'VCI - 3: 2D图形绘制',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/vci-ch03/";
            },},{id: "notes-vci-4-抗锯齿",
          title: 'VCI - 4: 抗锯齿',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/vci-ch04/";
            },},{id: "notes-vci-5-曲线",
          title: 'VCI - 5: 曲线',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/vci-ch05/";
            },},{id: "notes-数据结构与算法-3-栈与队列",
          title: '数据结构与算法 - 3: 栈与队列',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/dsa-ch03/";
            },},{id: "notes-ml-2-logistic-regression-逻辑回归",
          title: 'ML - 2: Logistic Regression (逻辑回归)',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/ML-ch02/";
            },},{id: "notes-ai数学基础-第3-4讲-随机变量",
          title: 'AI数学基础 - 第3-4讲: 随机变量',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/aimath-ch04/";
            },},{id: "notes-cv-2-图像形成-image-formation",
          title: 'CV - 2: 图像形成 (Image Formation)',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/cv-ch02/";
            },},{id: "notes-cv-3-图像处理-image-processing",
          title: 'CV - 3: 图像处理 (Image Processing)',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/cv-ch03/";
            },},{id: "notes-人工智能中的编程-第3章-并行通信-parallel-communication",
          title: '人工智能中的编程 - 第3章: 并行通信（Parallel Communication）',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/AIP-ch03/";
            },},{id: "notes-cv-4-特征检测-feature-detection",
          title: 'CV - 4: 特征检测 (Feature Detection)',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/cv-ch04/";
            },},{id: "notes-人工智能中的编程-第4章-并行算法-parallel-algorithms",
          title: '人工智能中的编程 - 第4章: 并行算法（Parallel Algorithms）',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/AIP-ch04/";
            },},{id: "notes-人工智能中的编程-第5章-并行算法ii-parallel-algorithms-ii",
          title: '人工智能中的编程 - 第5章: 并行算法II（Parallel Algorithms II）',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/AIP-ch05/";
            },},{id: "notes-数据结构与算法-第4章-字符串",
          title: '数据结构与算法 - 第4章: 字符串',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/dsa-ch04/";
            },},{id: "notes-人工智能中的编程-第6章-矩阵乘法-matrix-product",
          title: '人工智能中的编程 - 第6章: 矩阵乘法（Matrix Product）',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/AIP-ch06/";
            },},{id: "notes-cv-5-图像拼接-image-stitching",
          title: 'CV - 5: 图像拼接 (Image Stitching)',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/cv-ch05/";
            },},{id: "notes-数据结构与算法-第5章-二叉树-binary-tree",
          title: '数据结构与算法 - 第5章: 二叉树（Binary Tree）',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/dsa-ch05/";
            },},{id: "notes-cv-6-相机标定-camera-calibration",
          title: 'CV - 6: 相机标定 (Camera Calibration)',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/cv-ch06/";
            },},{id: "notes-cv-7-对极几何-epipolar-geometry",
          title: 'CV - 7: 对极几何 (Epipolar Geometry)',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/cv-ch07/";
            },},{id: "notes-vci-6-图像表示与处理",
          title: 'VCI - 6: 图像表示与处理',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/vci-ch06/";
            },},{id: "notes-vci-7-几何表示",
          title: 'VCI - 7: 几何表示',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/vci-ch07/";
            },},{id: "notes-vci-8-几何处理",
          title: 'VCI - 8: 几何处理',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/vci-ch08/";
            },},{id: "notes-cv-8-双目立体视觉-stereo-vision",
          title: 'CV - 8: 双目立体视觉 (Stereo Vision)',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/cv-ch08/";
            },},{id: "notes-数据结构与算法-第6章-树-tree",
          title: '数据结构与算法 - 第6章: 树（Tree）',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/dsa-ch06/";
            },},{id: "notes-人工智能中的编程-第7章-卷积和池化-convolution-and-pooling",
          title: '人工智能中的编程 - 第7章: 卷积和池化（Convolution and Pooling）',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/AIP-ch07/";
            },},{id: "notes-人工智能中的编程-第8章-pybind与单元测试-pybind-and-unit-test",
          title: '人工智能中的编程 - 第8章: Pybind与单元测试（Pybind and Unit Test）',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/AIP-ch08/";
            },},{id: "notes-人工智能中的编程-第9章-自动微分-automatic-differentiation",
          title: '人工智能中的编程 - 第9章: 自动微分（Automatic Differentiation）',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/AIP-ch09/";
            },},{id: "notes-cv-9-结构运动-structure-from-motion",
          title: 'CV - 9: 结构运动（Structure from Motion）',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/cv-ch09/";
            },},{id: "notes-vci-10-几何重建-geometry-reconstruction",
          title: 'VCI - 10: 几何重建（Geometry Reconstruction）',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/vci-ch10/";
            },},{id: "notes-数据结构与算法-第7章-图-graph",
          title: '数据结构与算法 - 第7章: 图（Graph）',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/dsa-ch07/";
            },},{id: "notes-vci-11-几何变换-geometric-transformations",
          title: 'VCI - 11: 几何变换 (Geometric Transformations)',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/vci-ch11/";
            },},{id: "notes-vci-12-高级渲染-advanced-rendering",
          title: 'VCI - 12: 高级渲染 (Advanced Rendering)',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/vci-ch12/";
            },},{id: "notes-vci-13-着色-shading",
          title: 'VCI - 13: 着色 (Shading)',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/vci-ch13/";
            },},{id: "notes-vci-14-渲染管线-graphics-pipeline",
          title: 'VCI - 14: 渲染管线 (Graphics Pipeline)',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/vci-ch14/";
            },},{id: "notes-人工智能中的编程-第10章-计算图-computational-graph",
          title: '人工智能中的编程 - 第10章: 计算图（Computational Graph）',
          description: "",
          section: "Notes",handler: () => {
              window.location.href = "/notes/AIP-ch10/";
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
        id: 'social-wechat_qr',
        title: 'Wechat_qr',
        section: 'Socials',
        handler: () => {
          window.open("", "_blank");
        },
      },{
        id: 'social-bilibili',
        title: 'Bilibili',
        section: 'Socials',
        handler: () => {
          window.open("https://space.bilibili.com/442187302", "_blank");
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
