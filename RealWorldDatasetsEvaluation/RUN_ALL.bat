@echo off
cd /d "%~dp0.."
python RealWorldDatasetsEvaluation\001run_all_real_world_experiments.py --continue-on-error
