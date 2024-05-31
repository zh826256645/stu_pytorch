import requests

for i in range(10):
    url = 'https://my.dianshangyi.com/validcode/captcha.gif'
    response = requests.get(url)
    if response.status_code == 200:
        with open(f"./data/test2/{i}.gif", 'wb') as f:
            f.write(response.content)
