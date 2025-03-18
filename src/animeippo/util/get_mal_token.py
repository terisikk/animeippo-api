import json
import os
import secrets

import dotenv
import requests

dotenv.load_dotenv("conf/prod.env")

CLIENT_ID = os.environ.get("MAL_CLIENT_ID")
CLIENT_SECRET = os.environ.get("MAL_CLIENT_SECRET")

MAL_API_URL = "https://myanimelist.net/v1"


# 1. Generate a new Code Verifier / Code Challenge.
def get_new_code_verifier() -> str:
    token = secrets.token_urlsafe(100)
    return token[:128]


# 2. Print the URL needed to authorise your application.
def print_new_authorisation_url(code_challenge: str):
    parameters = f"response_type=code&client_id={CLIENT_ID}&code_challenge={code_challenge}"
    url = f"{MAL_API_URL}/oauth2/authorize?{parameters}"
    print(f"Authorise your application by clicking here: {url}\n")


# 3. Once you've authorised your application, you will be redirected to the webpage you've
#    specified in the API panel. The URL will contain a parameter named "code" (the Authorisation
#    Code). You need to feed that code to the application.
def generate_new_token(authorisation_code: str, code_verifier: str) -> dict:
    url = f"{MAL_API_URL}/oauth2/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code": authorisation_code,
        "code_verifier": code_verifier,
        "grant_type": "authorization_code",
    }

    response = requests.post(url, data, timeout=10)
    response.raise_for_status()  # Check whether the request contains errors

    token = response.json()
    response.close()
    print("Token generated successfully!")

    with open("token.json", "w") as file:
        json.dump(token, file, indent=4)
        print('Token saved in "token.json"')

    return token


# 4. Test the API by requesting your profile information
def print_user_info(access_token: str):
    url = f"{MAL_API_URL}/users/@me"
    response = requests.get(url, headers={"Authorization": f"Bearer {access_token}"}, timeout=10)

    response.raise_for_status()
    user = response.json()
    response.close()

    print(f"\n>>> Greetings {user['name']}! <<<")


if __name__ == "__main__":
    code_verifier = code_challenge = get_new_code_verifier()
    print_new_authorisation_url(code_challenge)

    authorisation_code = input("Copy-paste the Authorisation Code: ").strip()
    token = generate_new_token(authorisation_code, code_verifier)

    print_user_info(token["access_token"])
