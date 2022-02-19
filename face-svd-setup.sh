python3 -m venv --system-site-packages ./venv
source ./venv/bin/activate  # sh, bash, or zsh
pip install --upgrade pip
#pip list  # show packages installed within the virtual environment
pip install -r requirements.txt
python face-svd.py
