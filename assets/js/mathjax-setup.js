window.MathJax = {
  tex: {
    tags: "ams",
    inlineMath: [
      ["$", "$"],
      ["\\(", "\\)"],
    ],
    displayMath: [
      ["$$", "$$"],
      ["\\[", "\\]"],
    ],
    processEscapes: true,
    processEnvironments: true,
    packages: { "[+]": ["ams", "newcommand", "configmacros"] },
    macros: {
      // Define custom macros if needed
    },
  },
  options: {
    skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"],
    includeHtmlTags: {
      br: "\n",
      wbr: "",
      "#comment": "",
    },
    renderActions: {
      addCss: [
        200,
        function (doc) {
          const style = document.createElement("style");
          style.innerHTML = `
          .mjx-container {
            color: inherit;
          }
        `;
          document.head.appendChild(style);
        },
        "",
      ],
    },
  },
  startup: {
    ready: function () {
      MathJax.startup.defaultReady();
      console.log("MathJax is ready!");
    },
  },
};
