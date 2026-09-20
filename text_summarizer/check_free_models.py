import requests
r = requests.get('https://openrouter.ai/api/v1/models')
models = r.json()['data']
free = [m for m in models if ':free' in m['id']]
for m in free[:20]:
    print(f"{m['id']}  context: {m.get('context_length', 'N/A')}")
