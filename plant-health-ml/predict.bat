@echo off
setlocal
set "P=%~1"
set "P=%P:\=/%"
wsl -d Ubuntu-24.04 bash -c "cd /mnt/c/Users/Lenovo/Desktop/ml-project/plant-health-ml && tfpython predict.py '%P%'"
