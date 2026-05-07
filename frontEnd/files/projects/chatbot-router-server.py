import pandas as pd
import requests, time, urllib.parse
from bs4 import BeautifulSoup
from urllib.robotparser import RobotFileParser
import sys
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
import requests
import os


DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")

# Startup cache for pre-fetched pages
CACHED_PAGES = {}
CACHE_FILE = os.path.join(DATA_DIR, "cached_pages.json")

server = FastMCP("ACCESS–NAIRR Minimal Data Server")
server.system_prompt = """
You are an ACCESS & NAIRR classification assistant.

Your job:

1. Classify each user query as one of:
• ACCESS
• NAIRR
• BOTH — please clarify
• UNSURE — please clarify

2. To make this decision, YOU MUST:
• Call get_access_resources() to retrieve ACCESS resource names
• Call get_nairr_resources() to retrieve NAIRR resource names
• Call get_resource_urls() to retrieve known URLs for each resource
• Call get_program_urls() to retrieve official program documentation URLs
• Call find_resource_by_name() to search resources by partial name match
• All URL fetches are cached and cleaned; you do not need to worry about raw HTML
• You MUST NOT fetch or browse URLs using your own internal browsing. All URL fetches MUST be done via get_cached_page() or fetch_url().
• If you encounter a URL in text, you MUST NOT follow it automatically. You may follow it ONLY by calling get_cached_page() or fetch_url().
• You MUST NEVER call fetch_url() before calling get_cached_page(). If you do, it is an error.
• Call fetch_url() after looking at cached text if and only if a classification still cannot be made.

3. Use this investigation strategy:
a. Identify resource names, program terms, or keywords in the user query.
b. Check get_access_resources() and get_nairr_resources().
   - If it appears in both → BOTH — please clarify
   - If only in one → classify accordingly
c. Use find_resource_by_name() to resolve partial or ambiguous names.
   - When calling find_resource_by_name(), pay attention to the field `systems_found`.
   - If it contains more than one system → return "BOTH — please clarify".
   - If it contains only one system → classify accordingly.
   - Do not assume a single system just because it appears first in the list.
e. Inspect cached page text via get_cached_page() using texts from previous tool calls.
f. Inspect get_resource_urls() and get_program_urls() for authoritative references if cached pages are not informative.
g. Call fetch_url() on URLs only if get_cached_page() didn't provide enough information; the server returns cleaned main text.
h. If a resource or program appears in both systems → BOTH — please clarify.
i. If the evidence is insufficient → UNSURE — please clarify.

4. Never answer the user's question directly. Only classify the system.
5. Never guess without a tool call.
6. Always check get_cached_page() before fetch_url().

4. Directions for web-searching
a. You must NOT use any internal web browsing or search features.
b. You must NOT follow or rewrite links yourself.
c. You must NOT perform semantic web search, URL inference, autonomous crawling, or domain guessing.
d. The only allowed actions for gathering external information are MCP tool calls to the following
    tools:
    • get_cached_page(url)
    • fetch_url(url)
    • get_resource_urls()
    • get_program_urls()
e. Any attempt to inspect a URL without calling these tools is an error.
f. If the tools do not provide enough information to classify, respond with "UNSURE — please clarify".

5. Output instructions:
• Your final answer must be one of:
  - "ACCESS"
  - "NAIRR"
  - "BOTH — please clarify"
  - "UNSURE — please clarify"
• Do not provide any additional explanation or text.
"""

df_access = pd.read_csv(os.path.join(DATA_DIR, "access_resources.csv"))
df_nairr = pd.read_csv(os.path.join(DATA_DIR, "nairr_resources.csv"))
df_urls = pd.read_csv(os.path.join(DATA_DIR, "resource_urls.csv"))

df_program_urls = pd.read_csv(os.path.join(DATA_DIR, "program_urls.csv"))

# Build allowed URL and domain lists for safe web-fetching
ALLOWED_DOMAINS = ["access-ci.org", "nairrpilot.org"]

# Build allowed URLs from CSVs
ALLOWED_URLS = set()
for row in df_urls.to_dict(orient="records"):
    access_url = row.get("ACCESS URL")
    nairr_url = row.get("NAIRR URL")

    if isinstance(access_url, str) and access_url.strip():
        ALLOWED_URLS.add(access_url.strip())

    if isinstance(nairr_url, str) and nairr_url.strip():
        ALLOWED_URLS.add(nairr_url.strip())

for row in df_program_urls.to_dict(orient="records"):
    url = row.get("Url")
    if isinstance(url, str) and url.strip():
        ALLOWED_URLS.add(url.strip())

# Load cache from disk if exists
if os.path.exists(CACHE_FILE):
    import json
    with open(CACHE_FILE, "r") as f:
        CACHED_PAGES = json.load(f)
else:
    CACHED_PAGES = {}

def url_allowed(url: str) -> bool:
    """
    Smart-strict allowlist:
    1. Exact URLs listed in CSVs
    2. Any URL whose domain contains allowed domains
    3. Any URL discovered while crawling allowed domains (still restricted to allowed domains)
    """
    if url in ALLOWED_URLS:
        return True
    try:
        u = urllib.parse.urlparse(url)
        host = u.netloc.lower()
    except:
        return False

    # Rule 2 and 3: allow anything within allowed domains
    for domain in ALLOWED_DOMAINS:
        if domain in host:
            return True

    return False

@server.tool()
def get_access_resources():
    """
    Return a structured list of ACCESS resources for Claude to reason over.
    """
    return df_access.to_dict(orient="records")

@server.tool()
def get_nairr_resources():
    """
    Return a structured list of NAIRR resources for Claude to reason over.
    """
    return df_nairr.to_dict(orient="records")

@server.tool()
def find_resource_by_name(name: str):
    """
    Finds ACCESS or NAIRR resources by partial name match.
    Returns all systems where the resource exists.
    """
    name_lower = name.lower()
    results = []
    systems_found = set()
    for row in df_access.to_dict(orient="records"):
        if name_lower in row["ACCESS Resource"].lower():
            results.append({"system": "ACCESS", **row})
            systems_found.add("ACCESS")
    for row in df_nairr.to_dict(orient="records"):
        if name_lower in row["NAIRR Resource"].lower():
            results.append({"system": "NAIRR", **row})
            systems_found.add("NAIRR")
    # Add a helper field to make overlap explicit
    results.append({"systems_found": list(systems_found)})
    return results


class URLInput(BaseModel):
    url: str = Field(..., description="URL to fetch")

@server.tool()
def get_resource_urls():
    return df_urls.to_dict(orient="records")

@server.tool()
def get_program_urls():
    """
    Returns URLs of official ACCESS and NAIRR program documentation.
    Claude uses this to investigate program-level terminology.
    """
    return df_program_urls.to_dict(orient="records")

@server.tool()
def get_cached_page(url: str):
    """Returns cached page text if present."""
    return {"url": url, "text": CACHED_PAGES.get(url, "")}

session = requests.Session()
session.headers.update({"User-Agent": "ACCESS-NAIRR-MCP/1.0"})

_host_last_fetch = {}
HOST_MIN_INTERVAL = 1.0  # seconds between requests to same host

def respect_robots(url):
    u = urllib.parse.urlparse(url)
    robots_url = f"{u.scheme}://{u.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(session.headers["User-Agent"], url)
    except:
        return True  # fail-open

def extract_main_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script","style","nav","footer","aside","header","form"]):
        s.decompose()
    main = soup.find("main")
    if main:
        return main.get_text("\n", strip=True)
    articles = soup.find_all("article")
    if articles:
        return max((a.get_text("\n", strip=True) for a in articles), key=len)
    return soup.get_text("\n", strip=True)

def fetch_url_clean(url):
    # Use cached if available
    if url in CACHED_PAGES:
        return {"url": url, "status": 200, "text": CACHED_PAGES[url][:12000], "snippet": CACHED_PAGES[url][:800]}
    if not url_allowed(url):
        return {"error": "URL not allowed by MCP server policy"}
    try:
        u = urllib.parse.urlparse(url)
        now = time.time()
        last = _host_last_fetch.get(u.netloc, 0)
        if now - last < HOST_MIN_INTERVAL:
            time.sleep(HOST_MIN_INTERVAL - (now - last))
        if not respect_robots(url):
            return {"error": "disallowed by robots.txt"}
        r = session.get(url, timeout=8)
        _host_last_fetch[u.netloc] = time.time()
        text = extract_main_text(r.text)
        return {"url": url, "status": r.status_code, "text": text[:12000], "snippet": text[:800]}
    except Exception as e:
        print(f"[ERROR] fetch_url {url}: {e}", file=sys.stderr)
        return {"error": str(e)}

# Pre-fetch allowed URLs on startup
def warm_cache():
    """
    Force-download and cache ALL allowed URLs at startup,
    independent of existing CACHED_PAGES entries.
    """
    global CACHED_PAGES
    updated_cache = {}

    for url in ALLOWED_URLS:
        try:
            # Fetch with live HTTP (ignoring existing cache)
            u = urllib.parse.urlparse(url)
            if not url_allowed(url):
                continue

            if not respect_robots(url):
                continue

            r = session.get(url, timeout=10)
            text = extract_main_text(r.text)
            updated_cache[url] = text[:20000]  # save more content
            print(f"[WARM_CACHE] Cached: {url}", file=sys.stderr)
        except Exception as e:
            print(f"[WARM_CACHE ERROR] {url}: {e}", file=sys.stderr)

    # Replace entire cache with freshly downloaded content
    CACHED_PAGES = updated_cache

    # Persist to disk
    import json
    with open(CACHE_FILE, "w") as f:
        json.dump(CACHED_PAGES, f)
    print(f"[WARM_CACHE] Completed with {len(CACHED_PAGES)} pages cached.", file=sys.stderr)

warm_cache()

if __name__ == "__main__":
    server.run()