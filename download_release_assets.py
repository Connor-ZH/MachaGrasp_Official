import argparse
import html
import os
import re
import sys
import tarfile
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

import requests


GOOGLE_DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1qb4T-0zB5JNEWcF9TGoJc1XEly7vAjff?usp=sharing"

ARCHIVES = {
    "checkpoints": {
        "filename": "machagrasp_checkpoints.tar.gz",
        "file_id": "1Cr4iw1VZ5dOSy2CTQCWz5AxVqZBzlyhx",
        "required": [
            "checkpoints/released_model.pth",
            "checkpoints/pretrained_pointnet_encoder.pth",
        ],
    },
    "data": {
        "filename": "machagrasp_data.tar.gz",
        "file_id": "1F-59ZioeIoVTTaMeoVsBGGeMTQvMXu6G",
        "required": [
            "data/allegro/train_2.pt",
            "data/barrett/train_2.pt",
            "data/shadow/train_2.pt",
            "data/graspnet/graspnet_meta.npz",
            "data/splits/unseen_object_list.json",
        ],
    },
    "assets": {
        "filename": "machagrasp_assets.tar.gz",
        "file_id": "11IWvBM_Jj8xGJpyjZTI_44hiwu_78qAF",
        "required": [
            "assets/allegro/allegro.urdf",
            "assets/barrett/barrett.urdf",
            "assets/shadow/shadow.urdf",
            "assets/objects/google_scanned_objects",
        ],
    },
}


def google_drive_url(file_id):
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def extract_google_drive_id(url):
    parsed = urlparse(url)
    if "drive.google.com" not in parsed.netloc:
        return None
    parts = parsed.path.strip("/").split("/")
    if "d" in parts:
        index = parts.index("d")
        if index + 1 < len(parts):
            return parts[index + 1]
    query = parse_qs(parsed.query)
    ids = query.get("id")
    return ids[0] if ids else None


def confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def drive_warning_form(response):
    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type:
        return None

    text = response.text
    form_match = re.search(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"', text)
    if not form_match:
        return None

    action = html.unescape(form_match.group(1))
    params = {}
    for name, value in re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', text):
        params[html.unescape(name)] = html.unescape(value)
    return urljoin(response.url, action), params


def ensure_archive_response(response, url):
    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type:
        raise RuntimeError(
            f"Google Drive returned an HTML page instead of an archive for {url}. "
            "Check that the file is shared publicly and that the URL/file id points to an archive file, not a folder."
        )


def download_file(url, output_path):
    file_id = extract_google_drive_id(url)
    if file_id:
        url = google_drive_url(file_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    response = session.get(url)
    token = confirm_token(response)
    if token:
        response = session.get(url, params={"confirm": token}, stream=True)
    else:
        warning_form = drive_warning_form(response)
        if warning_form:
            action, params = warning_form
            response = session.get(action, params=params, stream=True)
        else:
            response = session.get(url, stream=True)
    response.raise_for_status()
    ensure_archive_response(response, url)

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    downloaded = 0
    with tmp_path.open("wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            file.write(chunk)
            downloaded += len(chunk)
            print(f"\rdownloaded {output_path.name}: {downloaded / (1024 ** 2):.1f} MB", end="")
    print()
    tmp_path.replace(output_path)


def safe_extract(archive_path, root):
    root = root.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            target = (root / member.name).resolve()
            if not str(target).startswith(str(root) + os.sep):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
        archive.extractall(root)


def verify(root):
    missing = []
    for info in ARCHIVES.values():
        for relative_path in info["required"]:
            if not (root / relative_path).exists():
                missing.append(relative_path)
    if missing:
        raise FileNotFoundError("Missing required release files:\n" + "\n".join(missing))


def add_archive_args(parser, name):
    parser.add_argument(f"--{name}_url", default=None, help=f"Direct or Google Drive URL for {name} archive.")
    parser.add_argument(f"--{name}_file_id", default=None, help=f"Google Drive file id for {name} archive.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and extract MachaGrasp release files.",
        epilog=(
            "Release archive folder: "
            f"{GOOGLE_DRIVE_FOLDER_URL}. "
            "For manual download, place the archives in release_archives/ and run with --skip_download."
        ),
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--archive_dir", type=Path, default=None)
    parser.add_argument("--skip_download", action="store_true", help="Use existing archives in --archive_dir.")
    parser.add_argument("--no_extract", action="store_true", help="Download archives without extracting them.")
    parser.add_argument("--force", action="store_true", help="Redownload archives that already exist.")
    for name in ARCHIVES:
        add_archive_args(parser, name)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()
    archive_dir = args.archive_dir or root / "release_archives"
    archive_dir = archive_dir.resolve()

    for name, info in ARCHIVES.items():
        archive_path = archive_dir / info["filename"]
        url = getattr(args, f"{name}_url")
        file_id = getattr(args, f"{name}_file_id")

        if not args.skip_download and (args.force or not archive_path.exists()):
            if not file_id:
                file_id = info.get("file_id")
            if file_id:
                url = google_drive_url(file_id)
            if not url:
                raise ValueError(
                    f"Missing --{name}_url or --{name}_file_id. "
                    f"Expected archive name: {info['filename']}. "
                    f"Manual download folder: {GOOGLE_DRIVE_FOLDER_URL}"
                )
            download_file(url, archive_path)
        elif not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

        if not args.no_extract:
            print(f"extracting {archive_path.name} -> {root}")
            safe_extract(archive_path, root)

    if not args.no_extract:
        verify(root)
        print("release files are ready")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
