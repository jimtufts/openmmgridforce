#!/usr/bin/env python
import os
import sys
import shutil
import subprocess
from pathlib import Path

def clean_build():
    """Clean up build directories"""
    dirs_to_clean = ['_skbuild', 'dist', 'build', '*.egg-info']
    for dir_pattern in dirs_to_clean:
        for path in Path('.').glob(dir_pattern):
            if path.exists():
                if path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(path)

def build_package():
    """Build the package"""
    # Clean previous builds
    clean_build()
    
    # Build the wheel
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'build'])
    subprocess.check_call([sys.executable, '-m', 'build'])
    
    # Install the wheel
    wheels = list(Path('dist').glob('*.whl'))
    if wheels:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--force-reinstall', str(wheels[0])])

if __name__ == '__main__':
    build_package()
