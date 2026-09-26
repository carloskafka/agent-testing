"""One-time Gmail OAuth helper: mints a GOOGLE_REFRESH_TOKEN for your .env.

Run it after creating an OAuth 2.0 Client ID of type "Desktop app" in the Google
Cloud Console (https://console.cloud.google.com/apis/credentials). It opens a
browser, asks you to authorize read-only Gmail access, then prints a refresh
token you paste into .env.

Usage (either source the credentials from a file or from env vars):

  uv run python -m text_summarizer.gmail_oauth \
    --client-secret-file /path/to/client_secret.json [--console]

  # or, with GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET already in .env:
  uv run python -m text_summarizer.gmail_oauth [--console]

Use --console when the browser can't reach localhost back on this machine
(e.g. inside a VM): it prints a URL to open anywhere, and you paste the
authorization code back into the terminal.
"""

from __future__ import annotations

import argparse
import json
import os

import requests
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
TOKEN_URI = "https://oauth2.googleapis.com/token"


def _load_client_config(path: str | None) -> dict:
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise SystemExit(
            "Neither --client-secret-file nor GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET "
            "are set. See gmail_oauth.py module docstring."
        )
    return {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": TOKEN_URI,
            "redirect_uris": ["http://localhost"],
        }
    }


def _exchange_authorization_code(flow: InstalledAppFlow, code: str) -> dict:
    """POST the code to the token endpoint directly.

    ``google_auth_oauthlib``'s ``fetch_token`` raises when Google returns a
    different ``scope`` than requested (Google echoes every previously-granted
    scope for the client, e.g. Drive/Calendar). This bypasses that check so an
    already-consented token still lands in the response.
    """
    if flow.redirect_uri is None:
        raise RuntimeError("flow.redirect_uri is not set")
    cfg = flow.client_config
    client_id = cfg.get("client_id") or cfg["installed"]["client_id"]
    client_secret = cfg.get("client_secret") or cfg["installed"]["client_secret"]
    response = requests.post(
        TOKEN_URI,
        data={
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": flow.redirect_uri,
            "grant_type": "authorization_code",
            "code_verifier": flow.code_verifier or "",
        },
        timeout=60,
    )
    if response.status_code != 200:
        raise SystemExit(
            f"Token exchange failed (HTTP {response.status_code}): {response.text}"
        )
    token = response.json()
    if not token.get("refresh_token"):
        raise SystemExit(
            "No refresh_token in the token response. "
            f"Response was: {json.dumps(token)}"
        )
    return token


def main() -> None:
    parser = argparse.ArgumentParser(description="Mint a Gmail refresh token.")
    parser.add_argument(
        "--client-secret-file",
        default=None,
        help="Path to the client_secret.json downloaded from Google Cloud Console.",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Copy the printed URL into a browser and paste the code back into the "
        "terminal. Use this when the browser can't reach localhost (e.g. inside a VM).",
    )
    args = parser.parse_args()

    client_config = _load_client_config(args.client_secret_file)
    flow = InstalledAppFlow.from_client_config(client_config, scopes=SCOPES)

    if args.console:
        redirect_uri = client_config.get("installed", {}).get(
            "redirect_uris", ["http://localhost"]
        )[0]
        flow.redirect_uri = redirect_uri
        auth_url, _ = flow.authorization_url(
            access_type="offline", include_granted_scopes="false", prompt="consent"
        )
        print(
            "\nOpen this URL in any browser (e.g. on your host machine), sign in, "
            "approve read-only Gmail access, then copy the authorization code "
            "Google shows you and paste it below."
        )
        print(f"\n{auth_url}\n")
        try:
            code = input("Enter the authorization code: ").strip()
            token = _exchange_authorization_code(flow, code)
        except SystemExit as exc:
            raise
        except Exception as exc:  # pragma: no cover - library raises for expired codes
            raise SystemExit(
                f"Console flow failed: {exc}\n"
                "If the page said 'code expired', run this again and paste the "
                "new code promptly."
            )
    else:
        try:
            creds = flow.run_local_server(port=0, prompt="consent")
        except Exception as exc:  # pragma: no cover
            raise SystemExit(
                f"Local-server flow failed: {exc}\n"
                "If the browser can't reach localhost (VM?), use --console instead."
            )
        token = {
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "refresh_token": creds.refresh_token,
        }
        if not creds.refresh_token:
            raise SystemExit("No refresh token returned - did you grant access?")

    installed = client_config.get("installed", {})
    client_id = (
        installed.get("client_id")
        or creds.client_id
        or flow.client_config.get("client_id")
    )
    client_secret = (
        installed.get("client_secret")
        or creds.client_secret
        or flow.client_config.get("client_secret")
    )

    print("\n=== Add these to your text_summarizer/.env (never commit them) ===")
    print(f"GOOGLE_CLIENT_ID={client_id}")
    print(f"GOOGLE_CLIENT_SECRET={client_secret}")
    print(f"GOOGLE_REFRESH_TOKEN={token['refresh_token']}")


if __name__ == "__main__":
    main()