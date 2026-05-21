# 🚀 Serverless Python Shiny WebAssembly Dashboard for GitHub Pages

This directory contains a standalone, serverless, high-performance interactive dashboard built entirely using **Shiny for Python**, running inside the browser's WebAssembly environment via **Shinylive (Pyodide)**.

Because this dashboard is statically compiled and compiled client-side:
* **No Server Required**: Can be served entirely from any static file host, including **GitHub Pages**.
* **Zero Cost**: Completely free hosting on your custom domain `danielborek.me`.
* **Instant Sub-Second Reloads**: Once loaded, all Python packages and binaries are cached directly in the user's browser disk cache, enabling **instant sub-second reloads** on subsequent opens.

---

## 🛠️ Local Development & Modifying the App

To update the dashboard or make edits:

1. **Edit the Python Code**:
   Modify the main Shiny code inside [dashboard/app.py](file:///Users/daniel/PhD/spectral-comparison/code/dashboard/app.py) or dependencies in [dashboard/requirements.txt](file:///Users/daniel/PhD/spectral-comparison/code/dashboard/requirements.txt).

2. **Re-compile and Export the App**:
   Compile the changes and update the static WebAssembly bundle inside `docs/` by running:
   ```bash
   uv run shinylive export dashboard docs
   ```

3. **Run a Local Server to Test**:
   Test your dashboard locally before pushing:
   ```bash
   uv run python -m http.server --directory docs 8008
   ```
   Then open `http://localhost:8008` in your browser.

---

## 🌐 Deploying to GitHub Pages

Ensure that your compiled `docs/` files are pushed to your remote repository:

```bash
git add dashboard/app.py dashboard/requirements.txt docs/index.html docs/app.json docs/shinylive-sw.js
git commit -m "deploy serverless Python Shiny WASM dashboard"
git push origin presentation
```

---

## ⚙️ GitHub Pages & Custom Domain Setup

1. Go to your repository settings on GitHub.
2. Under **Pages**:
   - **Source**: Select `Deploy from a branch`.
   - **Branch**: Select your presentation/main branch, and choose the `/docs` folder.
   - Click **Save**.
3. Under **Custom Domain**:
   - The app will automatically serve from your main domain at:
     `https://danielborek.me/code-spectral-comparison/`

---

## 📱 Share via QR Code

Generate a high-quality vector (SVG) or high-resolution (PNG) QR code for:
* **Conference Posters**: Under the "Aperiodic Modeling" or "Methods" section, with a caption: *"Scan to play with the parameters dynamically in your browser!"*
* **Presentation Slides**: In the methodology or final slide of your talk.
* **Academic Papers**: Included as a figure or in the supplementary materials section.

---

## 💡 How does it work?

* **Pyodide WASM Engine**: Runs the official Python interpreter in the browser.
* **PyPI Wheel Bundling**: During compilation, `shinylive` automatically downloads Pyodide-compatible wheels for `numpy`, `scipy`, `matplotlib`, `specparam`, `shinyswatch`, and `shiny`, packaging them directly.
* **Offline Caching**: Built-in service workers (`shinylive-sw.js`) cache all compiled libraries, ensuring that after the first load, the site loads instantly.
* **Bootstrap Minty Theme**: Uses the `shinyswatch` package to apply a beautiful, professional, and accessible science-themed CSS stylesheet.
