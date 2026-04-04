from pathlib import Path

def find_project_root(start_path=None):
    """Find the project root by looking for pyproject.toml file."""
    if start_path is None:
        start_path = Path(__file__).parent

    current_path = Path(start_path).resolve()

    # Walk up the directory tree looking for pyproject.toml
    for parent in [current_path] + list(current_path.parents):
        if (parent / "pyproject.toml").exists():
            return parent

    # If not found, return current directory
    return current_path

def output():
    """Return the project output directory."""
    project_root = find_project_root()
    return project_root / "output"