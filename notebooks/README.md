# Notebooks

This folder holds runnable workflow notebooks for local experimentation.

Jupyter is optional and is not installed by the core project dependencies. To
open the notebooks from this repo:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[notebook,neural]"
.venv\Scripts\python.exe -m jupyter lab
```

Then open `notebooks/nursery_pathfinder_workflow.ipynb`.

The notebook is designed to run commands from the repository root even when the
browser starts inside `notebooks/`.
