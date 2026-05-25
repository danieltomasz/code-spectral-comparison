.ONESHELL:

# dashboard target name collides with the dashboard/ directory; mark phony
.PHONY: dashboard dashboard-serve dashboard-clean

PROJECT?=spectral-comparison
VERSION?=3.14
VENV=${PROJECT}-${VERSION}


sync:
	@echo "Syncing source code to uv workspace"
	uv sync --upgrade --all-extras


test:
	@echo "Running tests with uv"
	uv run pytest tests/

kernel:
	@echo "Installing Jupyter kernel"
	# Use the uv‑managed Python interpreter to register the kernel
	uv run python -m ipykernel install \
	    --user \
	    --name=${VENV} \
	    --display-name=${VENV}

context-py:
	files-to-prompt . -e py -e md  -e toml  --ignore  ./_archive/  ./.venv/ --cxml -o py-context.txt 

context-pesco:
	files-to-prompt ./pesco -e py -e md -e toml --ignore ./_archive/ ./.venv/ --cxml -o pesco-context.txt

ch2:
	quarto render chapters/ch02.qmd --to typst

preview:
	quarto preview chapters/ch02.qmd 	

docs:
	 uv run great-docs build && great-docs preview

# Export the Shiny app to a static Pyodide/WASM bundle in docs/ (committable)
dashboard:
	@echo "Exporting Shiny dashboard to static Pyodide bundle -> docs/"
	uv run shinylive export dashboard docs
	@echo "Stripping in-browser editor (no --no-editor flag in shinylive 0.8.8)"
	rm -rf docs/edit docs/shinylive/Editor.css docs/shinylive/Editor.js docs/shinylive/pyright
	@echo "Updating page title in docs/index.html"
	uv run python -c "p = 'docs/index.html'; c = open(p).read().replace('<title>Shiny App</title>', '<title>SpecParam Knee Simulation & Fitting Dashboard</title>'); open(p, 'w').write(c)"
	@echo "Done. Stage with: git add docs && git commit -m 'rebuild dashboard'"


dashboard-clean:
	@echo "Cleaning up generated Shinylive static files in docs/..."
	rm -rf docs/shinylive docs/app.json docs/index.html docs/shinylive-sw.js docs/edit



# Serve the exported bundle locally to verify before committing
dashboard-serve:
	@echo "Ensuring port 8008 is free..."
	@lsof -ti :8008 | xargs kill -9 2>/dev/null || true
	@echo "Opening Shiny app in your browser..."
	@(sleep 1 && open http://localhost:8008/) &
	uv run python -m http.server --directory docs 8008

MMDC_DIR := .cache/mmdc
MMDC_BIN := $(MMDC_DIR)/node_modules/.bin/mmdc
MMD_SRC  := $(wildcard diagrams/*.mmd)
MMD_PNG  := $(MMD_SRC:.mmd=.png)
MMD_SVG  := $(MMD_SRC:.mmd=.svg)

$(MMDC_BIN):
	@set -e; \
	echo "→ Installing @mermaid-js/mermaid-cli + puppeteer into $(MMDC_DIR)/ ..."; \
	mkdir -p $(MMDC_DIR); \
	cd $(MMDC_DIR); \
	[ -f package.json ] || npm init -y >/dev/null; \
	npm install --no-audit --no-fund @mermaid-js/mermaid-cli puppeteer

diagrams/%.png: diagrams/%.mmd $(MMDC_BIN)
	@set -e; \
	echo "→ Rendering $< -> $@"; \
	$(MMDC_BIN) -i $< -o $@ -b white -w 1600

diagrams/%.svg: diagrams/%.mmd $(MMDC_BIN)
	@set -e; \
	echo "→ Rendering $< -> $@"; \
	$(MMDC_BIN) -i $< -o $@ -b transparent

render-diagrams: $(MMD_PNG) $(MMD_SVG)
	@set -e; \
	if [ -z "$(MMD_SRC)" ]; then \
		echo "No diagrams/*.mmd files found."; \
	else \
		echo "✓ Rendered: $(MMD_PNG) $(MMD_SVG)"; \
	fi