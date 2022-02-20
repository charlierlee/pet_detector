python3 -m venv --system-site-packages ./venv
source ./venv/bin/activate  # sh, bash, or zsh
pip install --upgrade pip
pip install -U numpy
#sudo apt-get install python3-h5py
#sudo apt-get install libatlas-base-dev
#pip list  # show packages installed within the virtual environment
pip install -r rpi-requirements.txt

#pip install numpy --upgrade
source ./venv/bin/activate
python rpi-face-svd.py
