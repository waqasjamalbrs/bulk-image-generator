import io
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

import requests
import streamlit as st
import pandas as pd

API_URL_DEFAULT = "https://yousmind.com/api/image-generator/generate"
FILENAME_SAFE = re.compile(r"[^a-zA-Z0-9_\-]+")


def safe_name(s: str, max_len: int = 60) -> str:
    s = s.strip().replace(" ", "_")
    s = FILENAME_SAFE.sub("", s)
    return (s[:max_len] or "prompt").strip("_")


def download_image(url: str, timeout: int = 60) -> Optional[Tuple[str, bytes]]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()

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
    except Exception:
        return None


def build_zip(files: List[Tuple[str, bytes]]) -> bytes:
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

    # 1) Streamlit Secrets se key lene ki koshish (Streamlit Cloud / local .streamlit/secrets.toml)
    api_key = None
    try:
        api_key = st.secrets.get("YOUSMIND_API_KEY", None)
    except Exception:
        api_key = None

    # 2) Agar secrets me nahi hai to user se manga jayega
    if not api_key:
        api_key = st.text_input(
            "X-API-Key",
            type="password",
            help="Yousmind API key. Streamlit Cloud me Secrets me YOUSMIND_API_KEY ke naam se save karein.",
        )

    api_url = st.text_input("API URL", value=API_URL_DEFAULT)

    st.header("Generation Settings")

    provider = st.selectbox(
        "Provider",
        ["1.5-Fast", "1.0-Slow"],
        index=0,
        help="Docs ke mutabiq supported providers.",
    )

    aspect_ratio = st.selectbox(
        "Aspect ratio (image size)",
        ["16:9", "9:16", "1:1"],
        index=0,
        help='Docs: "16:9", "9:16", "1:1" supported.',
    )

    n_images = st.selectbox(
        "Images per prompt (n)",
        [1, 2, 3, 4],
        index=0,
        help="Docs: n ∈ {1,2,3,4}",
    )

    timeout = st.slider("Request timeout (sec)", 10, 180, 60)

    max_workers = st.slider(
        "Parallel requests",
        1,
        10,
        4,
        help="Zyada parallel = tez, lekin rate-limit ka risk.",
    )

st.subheader("Prompts input")

col1, col2 = st.columns(2)

with col1:
    prompts_text = st.text_area(
        "One prompt per line",
        height=220,
        placeholder=(
            "Har line pe ek prompt likhein.\n"
            "e.g.\n"
            "A majestic lion with a crown of stars, deep space background, cinematic lighting\n"
            "A futuristic city skyline at night, synthwave style"
        ),
    )
    st.caption("Khali lines ignore ho jayengi.")

with col2:
    uploaded = st.file_uploader(
        "Ya CSV upload karein (column name: prompt)", type=["csv"]
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

    if df is not None and "prompt" in df.columns:
        prompts.extend([str(x) for x in df["prompt"].dropna().tolist()])

    if prompts_text.strip():
        for line in prompts_text.splitlines():
            line = line.strip()
            if line:
                prompts.append(line)

    seen = set()
    unique_prompts: List[str] = []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            unique_prompts.append(p)
    prompts = unique_prompts

    if not prompts:
        st.error("Koi prompt nahi mila. Text area ya CSV me 'prompt' column zaroor bharain.")
        st.stop()

    st.info(f"{len(prompts)} prompts queue ho gaye.")

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

        files: List[Tuple[str, bytes]] = []
        for k, url in enumerate(urls, start=1):
            dl = download_image(url, timeout=timeout)
            if not dl:
                continue
            ext, raw = dl
            name = f"{idx:03d}_{safe_name(prompt)}_{k}.{ext}"
            files.append((name, raw))

        if not files:
            raise RuntimeError(
                f"image_urls mili lekin download fail ho gaya for prompt #{idx}: {prompt[:80]}..."
            )

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
            f"Koi image download nahi hui. Failures: {len(failures)}. Neeche error details dekhein."
        )

    if failures:
        st.subheader("Failures (error details)")
        for f in failures[:100]:
            st.code(f)

    if all_files:
        st.subheader("Preview (pehli kuch images)")
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
