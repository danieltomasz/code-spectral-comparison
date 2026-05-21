# 🚀 WebAssembly Interactive Dashboard for GitHub Pages

This directory contains a standalone, serverless, WebAssembly-powered interactive dashboard built using **stlite** (Streamlit compiled to WASM). 

Anyone visiting this page will be able to play with the interactive sliders **directly in their browser**—no Python installation, backend server, or setup required! This makes it ideal for embedding in research papers, conference posters, slides, or academic blogs using a QR code.

---

## 📋 Step 1: Push to GitHub

Ensure that your `docs/index.html` file is committed and pushed to your remote GitHub repository:

```bash
git add docs/index.html
git commit -m "add WASM interactive simulation dashboard for GitHub Pages"
git push origin main
```

---

## 🌐 Step 2: Enable GitHub Pages

1. Go to your repository page on GitHub (e.g., `https://github.com/your-username/your-repo-name`).
2. Click on the **⚙️ Settings** tab at the top.
3. In the left-hand sidebar, scroll down to the "Code and automation" section and click on **Pages**.
4. Under **Build and deployment**:
   - **Source**: Select `Deploy from a branch`.
   - **Branch**: Select your default branch (e.g., `main` or `master`).
   - **Folder**: Select the `/docs` folder (this tells GitHub to host the contents of the `docs` directory instead of the repository root).
5. Click **Save**.
6. Wait 1-2 minutes. Refresh the page, and GitHub will display your live public URL at the top:
   > 🔗 *Your site is live at `https://<your-username>.github.io/<your-repo-name>/`*

---

## 📱 Step 3: Generate a QR Code

To share this interactive dashboard at a conference, on slides, or in a paper:

1. Copy your public GitHub Pages URL (e.g., `https://danieltomasz.github.io/code-spectral-comparison/`).
2. Go to a free, reliable, high-quality QR code generator such as:
   - [QR Code Generator (free)](https://www.qr-code-generator.com/)
   - [Adobe Express QR Code Generator](https://www.adobe.com/express/feature/image/qr-code-generator)
   - [Beaconstac QR Code Generator](https://www.beaconstac.com/qr-code-generator)
3. Enter your URL and choose a clean, vector format (like **SVG** or high-resolution **PNG**) for optimal print quality.
4. Download the QR code and add it directly to:
   - **Conference Posters**: Under the "Aperiodic Modeling" or "Methods" section, with a caption: *"Scan to play with the parameters dynamically in your browser!"*
   - **Presentation Slides**: In the methodology or final slide of your talk.
   - **Academic Papers**: Included as a figure or in the supplementary materials section.

---

## 💡 How does it work?

Unlike traditional Streamlit apps which require a running cloud server (e.g., Streamlit Community Cloud), this app is powered by **WebAssembly (Pyodide)**:
- When a user visits the URL, their browser downloads a lightweight, sandboxed Python runtime (`Pyodide`) directly into their device's memory.
- It then installs standard analytical packages (`numpy`, `scipy`, `matplotlib`) and the pure-Python `specparam` package directly in the browser's sandbox.
- The entire simulation and fitting loop runs purely on the client's CPU.
- **Benefits**: Zero hosting costs, zero backend servers, completely secure, and it will remain active forever without any server downtime!
