"""
Download large model artifacts that are intentionally not committed to Git.

Render runs this during the build step. Set ADVANCED_AUTOENCODER_URL to a direct
download URL for models/advanced_autoencoder.keras.
"""

import hashlib
import os
import sys
import urllib.request


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "advanced_autoencoder.keras")
MODEL_URL_ENV = "ADVANCED_AUTOENCODER_URL"
MODEL_SHA256_ENV = "ADVANCED_AUTOENCODER_SHA256"


def sha256sum(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists: {MODEL_PATH}")
        return

    model_url = os.environ.get(MODEL_URL_ENV)
    if not model_url:
        print(
            f"Missing {MODEL_PATH} and {MODEL_URL_ENV} is not set.\n"
            f"Set {MODEL_URL_ENV} in Render to a direct download URL for "
            "advanced_autoencoder.keras.",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Downloading advanced autoencoder model to {MODEL_PATH}...")
    urllib.request.urlretrieve(model_url, MODEL_PATH)

    expected_sha256 = os.environ.get(MODEL_SHA256_ENV)
    if expected_sha256:
        actual_sha256 = sha256sum(MODEL_PATH)
        if actual_sha256.lower() != expected_sha256.lower():
            os.remove(MODEL_PATH)
            print(
                "Downloaded model checksum mismatch.\n"
                f"Expected: {expected_sha256}\n"
                f"Actual:   {actual_sha256}",
                file=sys.stderr,
            )
            sys.exit(1)

    print("Advanced autoencoder model is ready.")


if __name__ == "__main__":
    main()
