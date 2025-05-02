import aiohttp


BASE_URL = "http://teste:3000/api/teste"

async def get_Json(endpoint):
    url = f"{BASE_URL}/{endpoint}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    print(f"Erro: Status {resp.status} em {url}")
    except Exception as e:
        print(f"Erro ao acessar {url}: {e}")
    return None


async def post_Json(endpoint, data):
    url = f"{BASE_URL}/{endpoint}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as resp:
                return await resp.json()
    except Exception as e:
        print(f"Erro ao fazer POST para {url}: {e}")
    return None
