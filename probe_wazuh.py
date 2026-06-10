from dotenv import load_dotenv
import os
import httpx

load_dotenv()
user = os.getenv('WAZUH_USERNAME', 'wazuh')
pw = os.getenv('WAZUH_PASSWORD', 'wazuh')
base = os.getenv('WAZUH_BASE_URL', 'https://localhost:55000')

print('Base URL:', base)
with httpx.Client(verify=False) as c:
    r = c.get(base)
    print('ROOT', r.status_code)
    print(r.text[:2000])
    auth = c.post(base + '/security/user/authenticate', auth=(user, pw))
    print('AUTH', auth.status_code)
    try:
        print(auth.json())
    except Exception:
        print(auth.text)
    r2 = c.get(base + '/alerts')
    print('/alerts', r2.status_code)
    print(r2.text[:4000])
