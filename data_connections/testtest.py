from src.tests.inference_test import test_inference
from src.tests.load_test import test_model_loading, test_multi_Frame_model_loading, set_multi_frame_model, set_single_frame_model
import asyncio

async def run_tests():
    # retults = await test_model_loading()
    # retulst = await test_multi_Frame_model_loading()
    retults = await set_single_frame_model()
    retulst = await set_multi_frame_model()
    results = await test_inference(20)

asyncio.run(run_tests())  # Start the async event loop
