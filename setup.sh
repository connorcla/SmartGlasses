# Comment/Uncomment dependencies needed
# - Python : Programming Language (User Dependent)
# - p7zip-full : Compression/Unzip (User Dependent)
# - OLED_MODULE_Code : Examples of OLED Code Per Hardware && Language (Already Here)


# Update current libraries
sudo apt-get update

# Python

  # Linux
  sudo apt-get install python3-pip
  sudo apt-get install python3-pil
  sudo apt-get install python3-numpy
  sudo pip3 install spidev
  sudo apt-get install python3-smbus
  sudo apt-get install p7zip-full
  
  # Windows/Others?
  pip install numpy
  pip install pandas


# Matplotlib
python -m pip install -U pip
python -m pip install -U matplotlib

# Datasets
pip install kagglehub

# Pytorch

  # CPU/GPU
    # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

  # CPU ONLY
    pip3 install torch torchvision torchaudio

  # Mobile Pytorch (Don't know if necessary, also just guessed library)
  # pip3 install tqdm 
