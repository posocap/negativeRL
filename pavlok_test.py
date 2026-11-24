#!/usr/bin/env python3
"""Send ONE low-strength zap to your Pavlok to verify the API key. Needs --yes."""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

URL = "https://api.pavlok.com/api/v5/stimulus/send"


def send(token, strength, scheme, opener=urllib.request.urlopen):
    auth = f"{scheme} {token}".strip()
    body = json.dumps({"stimulus": {"stimulusType": "zap", "stimulusValue": strength,
                                    "reason": "api test"}}).encode()
    req = urllib.request.Request(URL, data=body, method="POST",
                                 headers={"Authorization": auth, "Content-Type": "application/json"})
    try:
        with opener(req, timeout=10) as r:
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def run(token, strength, opener=urllib.request.urlopen):
    status, body = send(token, strength, "Bearer", opener)
    if status == 401:
        print("401 with 'Bearer <key>'; retrying with bare key", flush=True)
        status, body = send(token, strength, "", opener)
    return status, body


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strength", type=int, default=10, help="1-50 (default 10)")
    p.add_argument("--yes", action="store_true", help="required: confirms you want a real zap")
    a = p.parse_args()
    if not a.yes:
        sys.exit("Refusing: this sends a REAL zap to the band. Re-run with --yes.")
    token = os.environ.get("PAVLOK_TOKEN")
    if not token:
        sys.exit("Set PAVLOK_TOKEN")
    status, body = run(token, max(1, min(a.strength, 50)))
    print(status, body)
    sys.exit(0 if 200 <= status < 300 else 1)


if __name__ == "__main__":
    main()
