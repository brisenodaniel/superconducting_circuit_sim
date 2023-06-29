from pathlib import Path
import os


def add_module_path() -> str:
    project_root = Path(__file__).parents[1]
    src_path = os.path.join(project_root, 'src')
    return src_path
