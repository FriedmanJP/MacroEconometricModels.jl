// Sun/Moon theme toggle for Solarized Light/Dark
// Respects system preference (prefers-color-scheme) on first visit;
// manual choice persisted in localStorage overrides system preference.
document.addEventListener("DOMContentLoaded", function () {
    // Hide the gear (settings) icon
    var settingsBtn = document.getElementById("documenter-settings-button");
    if (settingsBtn) settingsBtn.style.display = "none";

    // Create toggle button
    var btn = document.createElement("a");
    btn.className = "docs-navbar-link";
    btn.href = "#";
    btn.title = "Toggle light/dark theme";
    btn.style.cursor = "pointer";
    btn.style.fontSize = "1.1rem";
    btn.style.padding = "0 0.4rem";

    function isDark() {
        return document.documentElement.className === "theme--documenter-dark";
    }

    function updateIcon() {
        // Sun icon when dark (click to go light), Moon icon when light (click to go dark)
        btn.innerHTML = isDark()
            ? '<span class="fa-solid fa-sun" style="color:#b58900"></span>'
            : '<span class="fa-solid fa-moon" style="color:#586e75"></span>';
    }

    function setTheme(themeName) {
        var sheets = document.styleSheets;
        for (var i = 0; i < sheets.length; i++) {
            var node = sheets[i].ownerNode;
            if (!node) continue;
            var name = node.getAttribute("data-theme-name");
            if (!name) continue;
            sheets[i].disabled = (name !== themeName);
        }

        // Set the html class for dark theme CSS selectors
        if (themeName === "documenter-dark") {
            document.documentElement.className = "theme--documenter-dark";
        } else {
            document.documentElement.className = "";
        }

        // Persist choice
        if (window.localStorage) {
            window.localStorage.setItem("documenter-theme", themeName);
        }

        updateIcon();
    }

    // Determine initial theme: localStorage > system preference > light default
    function getInitialTheme() {
        if (window.localStorage) {
            var stored = window.localStorage.getItem("documenter-theme");
            if (stored === "documenter-dark" || stored === "documenter-light") {
                return stored;
            }
        }
        // No stored preference — follow system
        if (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) {
            return "documenter-dark";
        }
        return "documenter-light";
    }

    // Apply initial theme
    setTheme(getInitialTheme());

    // Listen for system preference changes (only if user hasn't manually chosen)
    if (window.matchMedia) {
        window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function (e) {
            // Only auto-switch if no manual preference stored
            if (window.localStorage && window.localStorage.getItem("documenter-theme")) return;
            setTheme(e.matches ? "documenter-dark" : "documenter-light");
        });
    }

    btn.addEventListener("click", function (e) {
        e.preventDefault();
        setTheme(isDark() ? "documenter-light" : "documenter-dark");
    });

    // Insert into navbar (before the settings button or at the end of docs-right)
    var docsRight = document.querySelector(".docs-right");
    if (docsRight) {
        if (settingsBtn) {
            docsRight.insertBefore(btn, settingsBtn);
        } else {
            docsRight.appendChild(btn);
        }
    }

    updateIcon();
});
