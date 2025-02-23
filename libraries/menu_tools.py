from typing import List

class MenuOption:
  def __init__(self, menu_option_name: str):
    self.menu_option_name = menu_option_name
    self.InitExtension()
    
  def InitExtension(self):
    pass


class Menu:
  def __init__(self, menu_name: str):
    self.menu_name = menu_name
    
    self.menu_options: List[MenuOption] = []
    self.InitExtension()

  def InitExtension(self):
    pass
    
  def AddMenuOption(self, menu_option_name: str):
    self.menu_options.append(MenuOption(menu_option_name))
    
  def RemoveMenuOption(self, menu_option_name: str):
    for i in self.menu_options:
      if i.menu_option_name == menu_option_name:
        self.menu_options.Remove(i)
        return
      
  def PrintMenu(self):
    print("----------")
    print("Menu")
    print("----------")

    for i in range(len(self.menu_options)):
      print(f"{i}: {self.menu_options[i].menu_option_name}")


  