import io
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from urllib.parse import urljoin

import requests
import streamlit as st
import pandas as pd

# -------------------------------------------------
# Page config (should be near the top)
# -------------------------------------------------
st.set_page_config(page_title="Bulk Image Generator", layout="wide")

# Hidden API URL (not shown in the UI)
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
        raise RuntimeError(f"Download failed for URL: {url} | Reason: {e}")

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


# -------------------------------------------------
# Session state initialization
# -------------------------------------------------
if "images" not in st.session_state:
    # List of (name: str, data: bytes)
    st.session_state["images"] = []

if "failures" not in st.session_state:
    st.session_state["failures"] = []

if "last_summary" not in st.session_state:
    st.session_state["last_summary"] = ""


# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("Bulk Image Generator")

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
            "API Key",
            type="password",
            help="Store your key in Streamlit Secrets for production use.",
        )

    st.header("Generation Settings")

    provider = st.selectbox(
        "Provider",
        ["1.5-Fast", "1.0-Slow"],
        index=0,
        help="Choose which engine to use.",
    )

    aspect_ratio = st.selectbox(
        "Aspect ratio (image size)",
        ["16:9", "9:16", "1:1"],
        index=0,
        help="Select the output aspect ratio.",
    )

    n_images = st.selectbox(
        "Images per prompt (n)",
        [1, 2, 3, 4],
        index=0,
        help="How many images to generate for each prompt.",
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

# -------------------------------------------------
# Generation logic (only when button clicked)
# -------------------------------------------------
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

        # Use the hidden API URL constant
        r = requests.post(API_URL_DEFAULT, headers=headers, json=payload, timeout=timeout)

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

        # Fix relative URLs (e.g. "/generated_images/xyz.png")
        fixed_urls = []
        for url in urls:
            if url.lower().startswith("http"):
                full_url = url
            else:
                full_url = urljoin(API_URL_DEFAULT, url)
            fixed_urls.append(full_url)

        files: List[Tuple[str, bytes]] = []
        for k, full_url in enumerate(fixed_urls, start=1):
            ext, raw = download_image(full_url, timeout=timeout)
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

    # Save results in session_state so they persist across reruns (e.g. download button click)
    st.session_state["images"] = all_files
    st.session_state["failures"] = failures
    st.session_state["last_summary"] = (
        f"Generated {len(all_files)} images from {len(prompts)} prompts in {elapsed:.1f}s. "
        f"Failures: {len(failures)}"
    )

    if all_files:
        st.success(st.session_state["last_summary"])
    else:
        st.error(
            "No images were downloaded. All downloads failed.\n\n"
            "Scroll down to 'Failures (error details)' and look at the first error."
        )

# -------------------------------------------------
# Show last summary (persistent)
# -------------------------------------------------
if st.session_state["last_summary"]:
    st.info(f"Last run: {st.session_state['last_summary']}")

# -------------------------------------------------
# Show failures from session state
# -------------------------------------------------
if st.session_state["failures"]:
    st.subheader("Failures (error details)")
    for f in st.session_state["failures"][:100]:
        st.code(f)

# -------------------------------------------------
# Preview + selection + single download button
# -------------------------------------------------
images = st.session_state["images"]

if images:
    st.subheader("Generated images")
    st.caption("Select the images you want in the ZIP, then click Download.")

    # Default selection: all True on first load
    for idx, (name, data) in enumerate(images):
        key = f"select_{idx}"
        if key not in st.session_state:
            st.session_state[key] = True  # default checked

    # Grid preview with checkboxes
    cols = st.columns(3)
    for idx, (name, data) in enumerate(images):
        col = cols[idx % 3]
        with col:
            st.image(data, caption=name, use_container_width=True)
            st.checkbox("Select", key=f"select_{idx}")

    # Build list of selected files
    selected_files: List[Tuple[str, bytes]] = []
    for idx, (name, data) in enumerate(images):
        if st.session_state.get(f"select_{idx}", False):
            selected_files.append((name, data))

    st.divider()

    # Single download button: behaves based on selection
    if selected_files:
        zip_bytes = build_zip(selected_files)
        st.download_button(
            "Download as ZIP",
            data=zip_bytes,
            file_name="images.zip",
            mime="application/zip",
        )
    else:
        st.warning("Select at least one image to enable download.")

    # Optional: clear button
    if st.button("Clear images"):
        old_images = list(images)  # copy length first
        st.session_state["images"] = []
        st.session_state["failures"] = []
        st.session_state["last_summary"] = ""
        # Clear selection flags
        for idx in range(len(old_images)):
            key = f"select_{idx}"
            if key in st.session_state:
                del st.session_state[key]
        st.experimental_rerun()
