from src.UDL_Kili_Interface import KILIConverter
from src.UDL_KEY import UDL_SENSOR_TO_KILI_ID
import os

kili = KILIConverter()
print(os.getpid())


print("Downloading recent collects...")
kili.download_collects()
print("Completed downloading recent collects")
print("Downloading missed collects")
kili.download_undownloaded_collects()
print("Completed downloading missed collects")