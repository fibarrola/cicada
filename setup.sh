conda update -n base -c defaults conda
conda env create -f environment.yml
conda activate cicada7

pip install -r requirements.txt

