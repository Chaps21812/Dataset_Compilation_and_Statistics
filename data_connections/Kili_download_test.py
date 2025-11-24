from src.UDL_Kili_Interface import KILIConverter
from src.UDL_KEY import UDL_SENSOR_TO_KILI_ID
import os

kili = KILIConverter()
print(os.getpid())

metadata_properties = {
  "expStartTime": {
    "type": "date",
    "filterable": True,
    "visibleByLabeler": True,
    "visibleByReviewer": True,
  },
  # … other metadata fields
}
# for key,value in UDL_SENSOR_TO_KILI_ID.items():
#     kili.kili.update_properties_in_project(value, metadata_properties=metadata_properties)

kili.download_UDL_upload_kili()