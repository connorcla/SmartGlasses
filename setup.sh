
# Update current libraries
sudo apt-get update

# Get python basics

  # Linux
  sudo apt-get install python3-pip
  sudo apt-get install python3-pil
  sudo apt-get install python3-numpy
  sudo pip3 install spidev
  sudo apt-get install python3-smbus


  # Windows/Others?
  pip install numpy
  pip install pandas

# 7-zip 
sudo apt-get install p7zip-full

# Matplotlib (Universal)
python -m pip install -U pip
python -m pip install -U matplotlib

# Datasets
pip install kagglehub

# Get pytorch libraries (Universal?)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Get mobile pytorch (Don't know if necessary, also just guessed library)
# pip3 install tqdm 