# GeoComp
A Python package for comparing geometric shapes at angled projections

## Installation

**Using `python3-venv`**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e . # to install in interactive mode
```
Later, you just need to type `source venv/bin/activate` in the termial to activate the environment.
If you want to use the code inside Jupyter Notebooks, you may have to run
```bash
pip install ipykernel
```
in the terminal afterward as well.

## Contribution
If you wish to contribute, install `pre-commit` first, via the following
```bash
pre-commit install
```
Then, before committing to the repository, run
```bash
pre-commit run -a
```
every time to ensure correct formatting.