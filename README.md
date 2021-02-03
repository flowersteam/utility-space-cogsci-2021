# Intrinsic Rewards in Human Curiosity-Driven Exploration: An Empirical Study
## Code repository

The repository contains all the code used to model the data from the study, as well as all the data used. However, the repository is not well-maintained as of yet. It will remain in this shape until a decision is made for the corresponding paper submitted to the CogScie 2021 conference (may be earlier, if we find time to clean it up).

**Some general pointers**:
* The modeling code is in the modeling folder. The main file there is `fit_models.py` which imports important functions from `modeling/loc_utils.py`.
* The code for producing figures in the `figures` folder is provided in the `analyses/figures.ipynb` notebook.
* The code for processing the data from the behavioral task as well as from the computational modeling is in the `data_processing.ipynb` notebook.
* The `requirements.txt` was automatically generated using `pip freeze` in the virtual environment, so it might contain more than what's needed to reproduce our results. 