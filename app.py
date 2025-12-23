import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from urllib.parse import urljoin

import requests
import streamlit as st
import pandas as pd

# -------------------------------------------------
# Page config (must be at the top)
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
    # Each item: {"name": str, "data": bytes, "prompt": str}
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
    # API key from secrets or manual input
    api_key = None
    try:
        api_key = st.secrets.get("YOUSMIND_API_KEY", None)
    except Exception:
        api_key = None

    if not api_key:
        api_key = st.text_input(
            "API Key",
            type="password",
            help="Store this in Streamlit Secrets as YOUSMIND_API_KEY for production use.",
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

    def worker(idx: int, prompt: str):
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

        # Use hidden API URL
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

        files = []
        for k, full_url in enumerate(fixed_urls, start=1):
            ext, raw = download_image(full_url, timeout=timeout)
            name = f"{idx:03d}_{safe_name(prompt)}_{k}.{ext}"
            files.append({"name": name, "data": raw, "prompt": prompt})

        return files

    progress = st.progress(0)
    status = st.empty()

    images_list = []
    failures: List[str] = []

    import time as _time
    start = _time.time()
    total = len(prompts)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(worker, i + 1, p): (i + 1, p)
            for i, p in enumerate(prompts)
        }

        done = 0
        for fut in as_completed(futures):
            idx, prompt_value = futures[fut]
            try:
                files = fut.result()
                images_list.extend(files)
            except Exception as e:
                failures.append(f"Prompt #{idx}: {prompt_value[:80]}... | Error: {e}")
            done += 1
            progress.progress(done / total)
            status.write(
                f"Completed: {done}/{total} prompts | Images: {len(images_list)} | Failed: {len(failures)}"
            )

    elapsed = _time.time() - start

    st.session_state["images"] = images_list
    st.session_state["failures"] = failures
    st.session_state["last_summary"] = (
        f"Generated {len(images_list)} images from {len(prompts)} prompts in {elapsed:.1f}s. "
        f"Failures: {len(failures)}"
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
# Preview + per-image select + edit+regen + single download button
# -------------------------------------------------
images = st.session_state["images"]

if images:
    st.subheader("Generated images")
    st.caption("Select images you want in the ZIP, then click Download.")

    # "Select all" button – sets all checkboxes to True
    if st.button("Select all"):
        for idx in range(len(images)):
            st.session_state[f"select_{idx}"] = True

    cols = st.columns(3)

    for idx, img in enumerate(images):
        key_select = f"select_{idx}"
        key_edit_mode = f"edit_mode_{idx}"
        key_prompt_edit = f"prompt_edit_{idx}"
        key_edit_btn = f"edit_btn_{idx}"
        key_regen = f"regen_{idx}"

        # Default edit mode: False
        if key_edit_mode not in st.session_state:
            st.session_state[key_edit_mode] = False

        with cols[idx % 3]:
            # Image preview
            st.image(img["data"], caption=img["name"], use_container_width=True)

            # Always show current prompt (short preview)
            st.caption(f"Current prompt: {img.get('prompt', '')[:80]}")

            # Selection checkbox (default False until user / Select all changes it)
            st.checkbox("Select", key=key_select)

            # Row: Edit prompt button + Regenerate button (always visible)
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("Edit prompt", key=key_edit_btn):
                    # Toggle edit mode; when turning on, seed the editable prompt
                    st.session_state[key_edit_mode] = not st.session_state[key_edit_mode]
                    if st.session_state[key_edit_mode]:
                        st.session_state[key_prompt_edit] = img.get("prompt", "")
                    st.experimental_rerun()
            with col_btn2:
                # Regenerate button always visible
                regen = st.button("Regenerate", key=key_regen)

            # If edit mode is on, show text area for this image
            if st.session_state[key_edit_mode]:
                st.text_area(
                    "Edit prompt for this image",
                    key=key_prompt_edit,
                )

            # Handle regeneration logic
            if regen:
                # If in edit mode and we have an edited prompt, use that; otherwise use original prompt
                if st.session_state.get(key_edit_mode, False) and key_prompt_edit in st.session_state:
                    prompt_to_use = st.session_state[key_prompt_edit]
                else:
                    prompt_to_use = img.get("prompt", "")

                headers = {
                    "Content-Type": "application/json",
                    "X-API-Key": api_key,
                }
                payload = {
                    "prompt": prompt_to_use,
                    "aspect_ratio": aspect_ratio,
                    "provider": provider,
                    "n": 1,  # regenerate a single image for this slot
                }
                r = requests.post(API_URL_DEFAULT, headers=headers, json=payload, timeout=timeout)
                try:
                    data = r.json()
                except Exception:
                    data = {"raw_text": r.text}

                if r.status_code != 200:
                    st.error(f"Regenerate failed: HTTP {r.status_code} | {str(data)[:200]}")
                else:
                    urls = data.get("image_urls", [])
                    if not urls:
                        st.error(f"Regenerate failed: no image_urls in response: {str(data)[:200]}")
                    else:
                        url = urls[0]
                        if not url.lower().startswith("http"):
                            url = urljoin(API_URL_DEFAULT, url)
                        ext, raw = download_image(url, timeout=timeout)
                        # Update this image in-place
                        img["data"] = raw
                        img["name"] = f"regen_{idx:03d}_{safe_name(prompt_to_use)}.{ext}"
                        img["prompt"] = prompt_to_use
                        st.session_state["images"][idx] = img
                        st.experimental_rerun()

    # Build list of selected files for download
    selected_files: List[Tuple[str, bytes]] = []
    for idx, img in enumerate(images):
        if st.session_state.get(f"select_{idx}", False):
            selected_files.append((img["name"], img["data"]))

    st.divider()

    # Single download button – uses selection
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

    # Optional clear button
    if st.button("Clear images"):
        old_len = len(images)
        st.session_state["images"] = []
        st.session_state["failures"] = []
        st.session_state["last_summary"] = ""
        # Clear per-image states
        for idx in range(old_len):
            for prefix in ("select_", "prompt_edit_", "edit_mode_", "edit_btn_", "regen_"):
                key = f"{prefix}{idx}"
                if key in st.session_state:
                    del st.session_state[key]
        st.experimental_rerun()
