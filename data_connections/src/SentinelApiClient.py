# myclient.py
import requests

class SentinelAPIClient:
    def __init__(self, host="http://localhost", port=30501):
        self.base_url = f"{host}:{port}"

    def get(self, endpoint, params=None):
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, params=params)
        response.raise_for_status()  # raise error if not 2xx
        return response.json()

    async def post(self, endpoint, data=None):
        url = f"{self.base_url}{endpoint}"
        # response = requests.post(url, data=data, json=json)
        # payload = {"data": data}
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    async def post_string(self, endpoint, data=None):
        url = f"{self.base_url}{endpoint}"
        # response = requests.post(url, data=data, json=json)
        payload = {"load_model": data}
        response = requests.post(url, params=payload)
        response.raise_for_status()
        return response.json()