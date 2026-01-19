import requests

STS_URL = "https://sts.nordpoolgroup.com/connect/token"
BASE_URL = "https://data-api.nordpoolgroup.com"
ENDPOINT = "/api/v2/PowerSystem/ProductionForecasts/ByLocations"

SCOPE = "marketdata_api"


def get_token_password(username: str, password: str, basic_auth_b64: str) -> str:
    """
    Nord Pool FAQ shows a password-grant example with Basic auth + scope=marketdata_api.
    If your portal provides a different grant_type, we’ll adjust.
    """
    headers = {
        "Authorization": f"Basic {basic_auth_b64}",
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    data = {
        "grant_type": "password",
        "scope": SCOPE,
        "username": username,
        "password": password,
    }
    r = requests.post(STS_URL, headers=headers, data=data, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Token request failed HTTP {r.status_code}: {r.text[:500]}")
    token = r.json().get("access_token")
    if not token:
        raise RuntimeError(f"No access_token in response: {r.text[:500]}")
    return token


def get_token_client_credentials(client_id: str, client_secret: str) -> str:
    """
    If your portal gives you client_id/client_secret for Market Data API, this is cleaner.
    """
    data = {
        "grant_type": "client_credentials",
        "scope": SCOPE,
    }
    r = requests.post(
        STS_URL,
        data=data,
        auth=(client_id, client_secret),  # HTTP Basic
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Token request failed HTTP {r.status_code}: {r.text[:500]}")
    token = r.json().get("access_token")
    if not token:
        raise RuntimeError(f"No access_token in response: {r.text[:500]}")
    return token


def fetch_one_day(token: str, delivery_date: str, location: str = "SE3") -> dict:
    url = f"{BASE_URL}{ENDPOINT}"
    params = {
        "deliveryDate": delivery_date,  # e.g. "2025-12-19"
        "location": location,
        "market": "DayAhead",
    }
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    r = requests.get(url, params=params, headers=headers, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Data request failed HTTP {r.status_code}\nURL: {r.url}\nResp: {r.text[:500]}")
    return r.json()


if __name__ == "__main__":
    # --- Choose ONE auth method ---

    # Option A (client credentials) - fill in if the portal gives you these
    # token = get_token_client_credentials(client_id="...", client_secret="...")

    # Option B (password grant) - fill in if the portal/FAQ flow applies to your account
    # "basic_auth_b64" is the base64 of "client_id:client_secret"
    token = get_token_password(
        username="YOUR_NORDPOOL_USERNAME",
        password="YOUR_NORDPOOL_PASSWORD",
        basic_auth_b64="BASE64(client_id:client_secret)",
    )

    data = fetch_one_day(token, delivery_date="2025-12-19", location="SE3")
    print("SUCCESS. Top-level keys:", list(data.keys()))
    print("Example snippet:", str(data)[:800])
