# Comment/Uncomment dependencies needed
# - Python : Programming Language (User Dependent)
# - p7zip-full : Compression/Unzip (User Dependent)
# - OLED_MODULE_Code : Examples of OLED Code Per Hardware && Language (Already Here)

echo
echo "--------------------------------------------------"
echo "Installing Python..."
echo

  sudo apt-get update
  sudo apt-get install python3-pip
  sudo apt-get install python3-pil
  sudo apt-get install python3-numpy
  sudo pip3 install spidev
  sudo apt-get install python3-smbus

  echo
  echo "Finished installing Python."

echo
echo "--------------------------------------------------"
echo 

echo
echo "--------------------------------------------------"
echo "Installing TestDemo..."
echo

  sudo apt-get install p7zip-full
  # sudo wget https://files.waveshare.com/upload/2/2c/OLED_Module_Code.7z
  # 7z x OLED_Module_Code.7z
  # cd OLED_Module_Code/RaspberryPi

echo
echo "--------------------------------------------------"
echo 