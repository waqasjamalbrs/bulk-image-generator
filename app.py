import io
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import requests
import streamlit as st
import pandas as pd

API_URL_DEFAULT = "https://yousmind.com/api/image-generator/generate"
FILENAME_SAFE = re.compile(r"[^a-zA-Z0-9_\-]+")


def safe_name(s: str, max_len: int = 60) -> str:
    """Convert a prompt into a safe file name."""
    s = s.strip().replace(" ", "_")
    s = FILENAME_SAFE.sub("", s)
    return (s[:max_len] or "prompt").strip("_")


def download_image(url: str, timeout: int = 60) -> Tuple[str, bytes]:
    """
    Download an image from a URL.
    If anything goes wrong, raise an error with the exact reason.
    """
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        # This message will appear in the Failures section
        raise RuntimeError(f"Download failed for URL: {url} | Reason: {e}")

    # Guess file extension from URL, fall back to content-type
    ext = "png"
    lower = url.lower()
    if ".jpg" in lower or ".jpeg" in lower:
        ext = "jpg"
    elif ".gif" in lower:
        ext = "gif"
    elif ".webp" in lower:
        ext = "webp"
    else:
        ct = (r.headers.get("content-type") or "").lower()
        if "jpeg" in ct or "jpg" in ct:
            ext = "jpg"
        elif "gif" in ct:
            ext = "gif"
        elif "webp" in ct:
            ext = "webp"

    return ext, r.content


def build_zip(files: List[Tuple[str, bytes]]) -> bytes:
    """Create a ZIP file in memory from (filename, data) pairs."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files:
            z.writestr(name, data)
    return buf.getvalue()


# --------------- STREAMLIT UI ---------------

st.set_page_config(page_title="Yousmind Bulk Image Generator", layout="wide")
st.title("Yousmind Bulk Image Generator")


with st.sidebar:
    st.header("API Settings")

    # 1) Try to read the key from Streamlit secrets (for Streamlit Cloud)
    api_key = None
    try:
        api_key = st.secrets.get("YOUSMIND_API_KEY", None)
    except Exception:
        api_key = None

    # 2) If there is no secret configured, fall back to manual input
    if not api_key:
        api_key = st.text_input(
            "X-API-Key",
            type="password",
            help="Your Yousmind API key. On Streamlit Cloud, store it in Secrets as YOUSMIND_API_KEY.",
        )

    api_url = st.text_input("API URL", value=API_URL_DEFAULT)

    st.header("Generation Settings")

    provider = st.selectbox(
        "Provider",
        ["1.5-Fast", "1.0-Slow"],
        index=0,
        help="Supported providers from the Yousmind docs.",
    )

    aspect_ratio = st.selectbox(
        "Aspect ratio (image size)",
        ["16:9", "9:16", "1:1"],
        index=0,
        help='Supported values: "16:9", "9:16", "1:1".',
    )

    n_images = st.selectbox(
        "Images per prompt (n)",
        [1, 2, 3, 4],
        index=0,
        help="The number of images to generate for each prompt.",
    )

    timeout = st.slider("Request timeout (seconds)", 10, 180, 60)

    max_workers = st.slider(
        "Parallel requests",
        1,
        10,
        4,
        help="More parallel requests = faster, but higher chance of hitting rate limits.",
    )

st.subheader("Prompts input")

col1, col2 = st.columns(2)

with col1:
    prompts_text = st.text_area(
        "One prompt per line",
        height=220,
        placeholder=(
            "Write one prompt per line.\n"
            "Example:\n"
            "A majestic lion with a crown of stars, deep space background, cinematic lighting\n"
            "A futuristic city skyline at night, synthwave style"
        ),
    )
    st.caption("Empty lines are ignored.")

with col2:
    uploaded = st.file_uploader(
        "Or upload a CSV (column name: prompt)", type=["csv"]
    )
    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview:", df.head())
        except Exception as e:
            st.error(f"CSV read error: {e}")

st.divider()

run = st.button(
    "Generate Bulk Images",
    type="primary",
    disabled=not bool(api_key),
)

if run:
    prompts: List[str] = []

    # From CSV
    if df is not None and "prompt" in df.columns:
        prompts.extend([str(x) for x in df["prompt"].dropna().tolist()])

    # From textarea
    if prompts_text.strip():
        for line in prompts_text.splitlines():
            line = line.strip()
            if line:
                prompts.append(line)

    # Remove duplicates while keeping order
    seen = set()
    unique_prompts: List[str] = []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            unique_prompts.append(p)
    prompts = unique_prompts

    if not prompts:
        st.error("No prompts found. Add lines in the text area or a CSV with a 'prompt' column.")
        st.stop()

    st.info(f"{len(prompts)} prompts queued.")

    def worker(idx: int, prompt: str) -> List[Tuple[str, bytes]]:
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        }
        payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "provider": provider,
            "n": n_images,
        }

        r = requests.post(api_url, headers=headers, json=payload, timeout=timeout)

        try:
            data = r.json()
        except Exception:
            data = {"raw_text": r.text}

        if r.status_code != 200:
            raise RuntimeError(
                f"HTTP {r.status_code} for prompt #{idx}: {prompt[:80]}...\n"
                f"Response: {str(data)[:400]}"
            )

        urls = data.get("image_urls", [])
        if not urls:
            raise RuntimeError(
                f"No image_urls in response for prompt #{idx}: {prompt[:80]}...\n"
                f"Response: {str(data)[:400]}"
            )

        # For debugging: show URLs for the first prompt
        if idx == 1:
            st.write("Debug: image_urls for the first prompt:")
            for u in urls:
                st.write(u)

        files: List[Tuple[str, bytes]] = []
        for k, url in enumerate(urls, start=1):
            # If something goes wrong here, the error will include the URL and reason
            ext, raw = download_image(url, timeout=timeout)
            name = f"{idx:03d}_{safe_name(prompt)}_{k}.{ext}"
            files.append((name, raw))

        return files

    progress = st.progress(0)
    status = st.empty()

    all_files: List[Tuple[str, bytes]] = []
    failures: List[str] = []

    start = time.time()
    total = len(prompts)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(worker, i + 1, p): (i + 1, p)
            for i, p in enumerate(prompts)
        }

        done = 0
        for fut in as_completed(futures):
            idx, prompt = futures[fut]
            try:
                files = fut.result()
                all_files.extend(files)
            except Exception as e:
                failures.append(f"Prompt #{idx}: {prompt[:80]}... | Error: {e}")
            done += 1
            progress.progress(done / total)
            status.write(
                f"Completed: {done}/{total} prompts | Images: {len(all_files)} | Failed: {len(failures)}"
            )

    elapsed = time.time() - start

    if all_files:
        st.success(
            f"Done! {len(all_files)} images generated from {len(prompts)} prompts "
            f"in {elapsed:.1f}s. Failures: {len(failures)}"
        )
    else:
        st.error(
            "No images were downloaded. All downloads failed.\n\n"
            "Scroll down to 'Failures (error details)' and look at the first error. "
            "It will show the exact URL and the reason (timeout, SSL error, blocked, etc.)."
        )

    if failures:
        st.subheader("Failures (error details)")
        for f in failures[:100]:
            st.code(f)

    if all_files:
        st.subheader("Preview (first few images)")
        preview = all_files[:12]
        cols = st.columns(4)
        for i, (name, data) in enumerate(preview):
            with cols[i % 4]:
                st.image(data, caption=name, use_container_width=True)

        zip_bytes = build_zip(all_files)
        st.download_button(
            "Download all images as ZIP",
            data=zip_bytes,
            file_name="yousmind_images.zip",
            mime="application/zip",
        )
