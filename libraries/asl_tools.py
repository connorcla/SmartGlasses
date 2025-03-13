from libraries.nn_tools import *
import subprocess

class MainPathManager(DataPathManager):
  def InitExtension(self):
    
    self.AddDataPath("model_path", "./asl_model/")
    self.AddDataPath("image_path", "./camera_image/captured_image.jpg")

class NNDefaultTransform(NNTransform):
  def GetTransformation(self):
    transform = transforms.Compose([
      transforms.Resize(128),
      transforms.ToTensor()
    ])
    return transform
class NNDefaultTransformHigh(NNTransform):
  def GetTransformation(self):
    transform = transforms.Compose([
      transforms.Resize(224),
      transforms.ToTensor()
    ])
    return transform
class NNSquareCropTransform(NNTransform):
  def GetTransformation(self):
    transform = transforms.Compose([
      transforms.CenterCrop(1080),
      transforms.Resize(128),
      transforms.ToTensor()
    ])
    return transform 
class NNSquareCropTransformHigh(NNTransform):
  def GetTransformation(self):
    transform = transforms.Compose([
      transforms.CenterCrop(1080),
      transforms.Resize(224),
      transforms.ToTensor()
    ])
    return transform 

class NNASL(NNDefault):
  def InitExtension(self):
    self.AddTrainingAttributeGroup("Default", 0.2, 32, 100, 0.01, 29) 

    self.AddNNTransformation(NNDefaultTransform("Default"))
    self.AddNNTransformation(NNDefaultTransformHigh("DefaultHigh"))
    self.AddNNTransformation(NNSquareCropTransform("Square"))
    self.AddNNTransformation(NNSquareCropTransformHigh("SquareHigh"))

    self.AddNNModel(NNModel("Default", False))
    self.AddNNModel(NNModel("Mobile", True))
    

  
def ConvertLabelToChar(label: int):
  return ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',' ',' ',' '][label]

def CaptureImage(filename="camera_image/captured_image.jpg"):
  try:
      subprocess.run(["rpicam-still", "-o", filename], check=True)
      print("Image captured successfully")
  except subprocess.CalledProcessError:
      print("Error: Failed to capture image with rpicam-still.")
      exit()
