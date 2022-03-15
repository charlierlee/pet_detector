python3 -m venv --system-site-packages ./venv
source ./venv/bin/activate  # sh, bash, or zsh
pip install --upgrade pip
#pip list  # show packages installed within the virtual environment
pip install -r face-cv2-requirements.txt
source ./venv/bin/activate
python face-cv2.py
