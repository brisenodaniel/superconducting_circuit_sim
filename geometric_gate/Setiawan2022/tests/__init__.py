import sys
from pathlib import Path
import os
# setup path variables for scripts
project_root = Path(__file__).parents[1]
print(project_root)
# add src and test modules to path
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, './tests'))
# set cwd to project root
os.chdir(project_root)
# sys.path.append('./src')
# sys.path.append('./tests')
