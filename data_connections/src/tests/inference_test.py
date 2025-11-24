from ..raw_datset import raw_dataset
from ..SentinelApiClient import SentinelAPIClient
import time
import numpy as np

test_image_path = "/data/Dataset_Compilation_and_Statistics/Sentinel_Datasets/CalsatTesting_datasets/Calsat_Final-ABQ01-2025"


async def test_inference(num_collects:int):
    API_CLIENT = SentinelAPIClient()
    collect_list = raw_dataset(test_image_path).encode_fits_collects(num_collects)


    numbers = []
    per_image_time = []
    for data in collect_list:
        start = time.perf_counter()
        await API_CLIENT.post("/single_frame_inference",data)  # actually awaits the request
        end = time.perf_counter()
        avg_time = (end - start)
        numbers.append(avg_time)
        per_image_time.append(avg_time/len(data))
    print(f"Average Model Inference time: {np.average(numbers):.4f}")
    print(f"Average time per request: {np.average(numbers):.4f} s")
    print(f"Average time per image: {np.average(per_image_time):.4f} s")
    return avg_time

