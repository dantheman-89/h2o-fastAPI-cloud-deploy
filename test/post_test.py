import requests

url = 'http://127.0.0.1:8000/prediction'
request_obj = {'requestID': 'xhm0001','sepal_length_cm':'2.5','sepal_width_cm':'2.5','petal_length_cm':'2.5','petal_width_cm':'2.5' }

results = requests.post(url, json = request_obj)

print(results.text)