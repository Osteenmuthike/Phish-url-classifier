import re
import tldextract
from urllib.parse import urlparse

def extract_features(url):
    """
    Extract handcrafted features from a URL for phishing detection.
    Returns a dictionary of numerical values for ML models.
    """

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    ext = tldextract.extract(url)

    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url),
        "has_ip": 1 if re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", url) else 0,
        "https_present": 1 if parsed.scheme == "https" else 0,
        "num_subdomains": len(ext.subdomain.split(".")) if ext.subdomain else 0,
        "tld_length": len(ext.suffix),
        "contains_login": 1 if "login" in url.lower() else 0,
        "contains_secure": 1 if "secure" in url.lower() else 0,
        "contains_account": 1 if "account" in url.lower() else 0,
        "contains_update": 1 if "update" in url.lower() else 0,
        "contains_verify": 1 if "verify" in url.lower() else 0,
    }

    return features


if __name__ == "__main__":
    test_url = "https://secure-login-example.com/account/update"
    print(extract_features(test_url))
