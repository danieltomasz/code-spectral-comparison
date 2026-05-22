.ONESHELL:

# dashboard target name collides with the dashboard/ directory; mark phony
.PHONY: dashboard dashboard-serve

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
	@echo "Done. Stage with: git add docs && git commit -m 'rebuild dashboard'"

# Serve the exported bundle locally to verify before committing
dashboard-serve:
	uv run python -m http.server --directory docs 8008