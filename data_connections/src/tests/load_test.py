from ..SentinelApiClient import SentinelAPIClient
import timeit
import time
import numpy as np

async def set_single_frame_model(model = "LMNT01_E249.torchscript"):
    API = SentinelAPIClient()
    await API.post_string("/select_SF_model", model)  # actually awaits the request
    print(f"Posted single frame model {model}")

async def set_multi_frame_model(model = "LMNT01_E249.torchscript"):
    API = SentinelAPIClient()
    await API.post_string("/select_MF_model", model)  # actually awaits the request
    print(f"Posted single frame model {model}")

async def test_model_loading():
    MODEL_A = "LMNT01_E249.torchscript"
    MODEL_B = "LMNT01_E249recentroid.torchscript"
    MODEL_C = "LMNT02_E180.torchscript"
    MODEL_D = "LMNT02_E180recentroid.torchscript"
    MODEL_E = "RME04_E249recentroid.torchscript"
    API = SentinelAPIClient()
    num_attempt = 25

    registered_models = [MODEL_A,MODEL_B,MODEL_C,MODEL_D,MODEL_E] 

    for _ in range(num_attempt):
        numbers = []
        for model_string in registered_models:
            start = time.perf_counter()
            await API.post_string("/select_SF_model", model_string)  # actually awaits the request
            end = time.perf_counter()
            avg_time = (end - start)
            numbers.append(avg_time)
    print(f"Average Single frame Model Load time: {np.average(numbers):.4f}")
    return avg_time

async def test_multi_Frame_model_loading():
    MODEL_A = "LMNT01_E249.torchscript"
    MODEL_B = "LMNT01_E249recentroid.torchscript"
    MODEL_C = "LMNT02_E180.torchscript"
    MODEL_D = "LMNT02_E180recentroid.torchscript"
    MODEL_E = "RME04_E249recentroid.torchscript"
    API = SentinelAPIClient()
    num_attempt = 25

    registered_models = [MODEL_A,MODEL_B,MODEL_C,MODEL_D,MODEL_E] 

    for _ in range(num_attempt):
        numbers = []
        for model_string in registered_models:
            start = time.perf_counter()
            await API.post_string("/select_MF_model", model_string)  # actually awaits the request
            end = time.perf_counter()
            avg_time = (end - start)
            numbers.append(avg_time)
    print(f"Average multi frame Model Load time: {np.average(numbers):.4f}")
    return avg_time

