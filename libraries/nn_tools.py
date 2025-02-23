from typing import List
import os

class DataPath():
  def __init__(self, path_name: str, literal_path: str):
    self.path_name = path_name
    self.literal_path = literal_path

    self.InitExtension()

  def InitExtension(self):
    pass


class DataPathManager():
  def __init__(self, path_manager_name: str):
    self.path_manager_name = path_manager_name

    self.data_paths_list: List[DataPath] = []
    self.InitExtension()
    
  def InitExtension(self):
    pass
    
  def AddDataPath(self, path_name: str, literal_path: str):
    if not os.path.exists(literal_path):
        raise FileNotFoundError(f"Path ({literal_path}) is not valid.")
    self.data_paths_list.append(DataPath(path_name, literal_path))
    print(f"Added \'{path_name}\': {literal_path}")

  def RemoveDataPath(self, path_name):
    for i in self.data_paths_list:
      if (i.path_name == path_name):
        self.data_paths_list.remove(path_name)
        return
  
  def GetLiteralDataPath(self, path_name: str):
    for i in self.data_paths_list:
      if (i.path_name == path_name):
        return i.literal_path
    raise LookupError(f"\'{path_name}\' was not found in the data manager.")
    
    
class TrainingAttributes():
  def __init__ (self, test_size: float, batch_size: int, num_epoch: int, learning_rate: float, num_classes: int):
    self.test_size = test_size
    self.batch_size = batch_size
    self.num_epoch = num_epoch
    self.learning_rate = learning_rate
    self.num_classes = num_classes