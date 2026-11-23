"""Private R2 drive. Configure R2_* and DRIVE_PASSWORD_HASH on Render.

Independent owner-only login; existing employee sessions do not grant access.
"""
import hashlib
import hmac
import os
import secrets
import time
import threading
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from urllib.parse import quote
from io import BytesIO
from PIL import Image
import math

import boto3
from botocore.config import Config
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, Response, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

router = APIRouter(prefix="/storage", tags=["private-drive"])
PAGE = Path(__file__).with_name("storage_ui.html")
MAX_UPLOAD_BYTES = 5 * 1024**3
MAX_PROXY_UPLOAD_BYTES = 200 * 1024**2
_login_attempts = {}
_login_lock = threading.Lock()
REMEMBER_SECONDS = 30 * 24 * 60 * 60
REMEMBER_COOKIE = 'alcuin_drive_trusted'


def configured():
    return all(os.getenv(k) for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "DRIVE_PASSWORD_HASH"))


def bucket():
    return os.environ["R2_BUCKET"]


def client():
    return boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto", config=Config(signature_version="s3v4"))


def remember_signer():
    # Password changes invalidate all remembered devices.
    secret = os.environ.get("SESSION_SECRET", "")
    if not secret or secret == "change-this-in-render":
        return None
    binding = hashlib.sha256(os.environ.get("DRIVE_PASSWORD_HASH", "").encode()).hexdigest()
    return URLSafeTimedSerializer(secret_key=secret, salt="alcuin-drive-v1-" + binding)


def authorized(request):
    signer = remember_signer()
    token = request.cookies.get(REMEMBER_COOKIE)
    if signer and token:
        try:
            return signer.loads(token, max_age=REMEMBER_SECONDS) == "drive-owner"
        except (BadSignature, SignatureExpired):
            pass
    return request.session.get("private_drive_owner") is True


def check_origin(request):
    origin = request.headers.get('origin')
    if origin:
        from urllib.parse import urlparse
        parsed = urlparse(origin)
        if parsed.scheme != 'https' or parsed.netloc != request.headers.get('host'):
            raise HTTPException(403, 'Invalid request origin')


def require_owner(request):
    if not authorized(request):
        raise HTTPException(401, "Private drive login required")
    if not configured():
        raise HTTPException(503, "R2 drive not configured")


def clean_path(value):
    value = value.strip().strip("/")
    parts = value.split("/") if value else []
    if any(p in (".", "..") or "\\" in p or not p or any(ord(c) < 32 for c in p) for p in parts):
        raise HTTPException(400, "Invalid path")
    return "/".join(parts)


def verify_password(password):
    # Format: pbkdf2_sha256$iterations$salt_hex$hash_hex
    try:
        scheme, iterations, salt, expected = os.environ["DRIVE_PASSWORD_HASH"].split("$")
        if scheme != "pbkdf2_sha256" or not 100000 <= int(iterations) <= 2000000:
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations))
        return hmac.compare_digest(actual, bytes.fromhex(expected))
    except (ValueError, KeyError):
        return False


class Login(BaseModel):
    password: str
    remember: bool = True


class Upload(BaseModel):
    path: str
    content_type: str = "application/octet-stream"
    size: int = Field(ge=0, le=MAX_UPLOAD_BYTES)


class PathRequest(BaseModel):
    path: str


class MoveRequest(BaseModel):
    source: str
    destination: str


@router.get("")
def page():
    return FileResponse(PAGE, media_type="text/html")


@router.get("/api/status")
def status(request: Request):
    return {"authenticated": authorized(request), "configured": configured()}


@router.post("/api/login")
def login(request: Request, payload: Login):
    check_origin(request)
    address = request.client.host if request.client else 'unknown'
    now = time.monotonic()
    with _login_lock:
        if len(_login_attempts) > 10000:
            _login_attempts.clear()
        attempts = [t for t in _login_attempts.get(address, []) if now-t < 900]
        if len(attempts) >= 8:
            raise HTTPException(429, 'Too many attempts; retry later')
        _login_attempts[address] = attempts
    if not configured():
        raise HTTPException(503, "Drive not configured")
    if not verify_password(payload.password):
        with _login_lock:
            _login_attempts.setdefault(address, []).append(now)
        raise HTTPException(401, "Invalid credentials")
    with _login_lock:
        _login_attempts.pop(address, None)
    request.session["private_drive_owner"] = True
    response = JSONResponse({"ok": True, "remembered": payload.remember})
    if payload.remember:
        signer = remember_signer()
        if not signer:
            raise HTTPException(503, "Persistent login is not configured")
        response.set_cookie(REMEMBER_COOKIE, signer.dumps("drive-owner"),
                            max_age=REMEMBER_SECONDS, path="/storage",
                            secure=True, httponly=True, samesite="lax")
    else:
        response.delete_cookie(REMEMBER_COOKIE, path="/storage")
    return response


@router.post("/api/logout")
def logout(request: Request):
    check_origin(request)
    request.session.pop("private_drive_owner", None)
    response = JSONResponse({"ok": True})
    response.delete_cookie(REMEMBER_COOKIE, path="/storage", secure=True, httponly=True, samesite="lax")
    return response


@router.get("/api/files")
def files(request: Request, prefix: str = "", cursor: str = ""):
    require_owner(request)
    prefix = clean_path(prefix)
    prefix = prefix + "/" if prefix else ""
    args = {"Bucket": bucket(), "Prefix": prefix, "Delimiter": "/", "MaxKeys": 500}
    if cursor:
        args["ContinuationToken"] = cursor
    response = client().list_objects_v2(**args)
    items = []
    for p in response.get("CommonPrefixes", []):
        items.append({"path": p["Prefix"].rstrip("/"), "name": p["Prefix"][len(prefix):].rstrip("/"), "folder": True})
    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key.endswith("/"):
            continue
        items.append({"path": key, "name": key[len(prefix):], "folder": False, "size": obj["Size"],
            "modified": obj["LastModified"].isoformat()})
    return {"items": items, "next_cursor": response.get("NextContinuationToken")}


@router.post("/api/folder")
def folder(request: Request, payload: PathRequest):
    require_owner(request)
    check_origin(request)
    path = clean_path(payload.path)
    if not path:
        raise HTTPException(400, "Folder name required")
    client().put_object(Bucket=bucket(), Key=path + "/", Body=b"")
    return {"ok": True}


@router.post("/api/upload-url")
def upload_url(request: Request, payload: Upload):
    require_owner(request)
    check_origin(request)
    path = clean_path(payload.path)
    if not path or path.endswith("/"):
        raise HTTPException(400, "File path required")
    if payload.size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large")
    url = client().generate_presigned_url("put_object",
        Params={"Bucket": bucket(), "Key": path, "ContentType": payload.content_type},
        ExpiresIn=300)
    return {"url": url, "headers": {"Content-Type": payload.content_type}}


@router.get("/api/download-url")
def download_url(request: Request, path: str):
    require_owner(request)
    path = clean_path(path)
    if not path:
        raise HTTPException(400, "File required")
    url = client().generate_presigned_url("get_object",
        Params={"Bucket": bucket(), "Key": path}, ExpiresIn=120)
    return {"url": url}


@router.post("/api/delete")
def delete(request: Request, payload: PathRequest):
    require_owner(request)
    check_origin(request)
    path = clean_path(payload.path)
    if not path:
        raise HTTPException(400, "Cannot delete root")
    s3 = client()
    if path.endswith("/"):
        raise HTTPException(400, "Invalid path")
    # If folder, remove its contents in batches, including folder marker.
    listing = s3.list_objects_v2(Bucket=bucket(), Prefix=path + "/", MaxKeys=1)
    if listing.get("Contents"):
        prefix = path + "/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket(), Prefix=prefix):
            objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
            if objects:
                s3.delete_objects(Bucket=bucket(), Delete={"Objects": objects})
        s3.delete_object(Bucket=bucket(), Key=prefix)
    else:
        s3.delete_object(Bucket=bucket(), Key=path)
    return {"ok": True}


@router.post("/api/move")
def move(request: Request, payload: MoveRequest):
    require_owner(request)
    check_origin(request)
    source, dest = clean_path(payload.source), clean_path(payload.destination)
    if not source or not dest or source == dest:
        raise HTTPException(400, "Invalid source or destination")
    s3 = client()
    # Only individual objects; folder moves require an explicit separate operation.
    try:
        s3.head_object(Bucket=bucket(), Key=source)
    except Exception:
        raise HTTPException(404, "Source file not found")
    try:
        s3.head_object(Bucket=bucket(), Key=dest)
        raise HTTPException(409, "Destination already exists")
    except HTTPException:
        raise
    except Exception:
        pass
    s3.copy_object(Bucket=bucket(), Key=dest, CopySource={"Bucket": bucket(), "Key": source})
    s3.delete_object(Bucket=bucket(), Key=source)
    return {"ok": True}


@router.post("/api/upload")
def upload(request: Request, path: str = Form(...), file: UploadFile = File(...)):
    """Same-origin streaming upload; does not require permissive R2 bucket CORS."""
    require_owner(request)
    check_origin(request)
    key = clean_path(path)
    if not key:
        raise HTTPException(400, "File path required")
    # Spooled file is disk-backed above the Starlette threshold.
    file.file.seek(0, 2)
    size = file.file.tell()
    file.file.seek(0)
    if size > MAX_PROXY_UPLOAD_BYTES:
        raise HTTPException(413, "Upload exceeds 200 MB; use multipart upload for larger files")
    client().upload_fileobj(file.file, bucket(), key, ExtraArgs={"ContentType": file.content_type or "application/octet-stream"})
    return {"ok": True, "size": size}

@router.get("/api/search")
def search(request: Request, q: str = "", kind: str = "", start: str = "", end: str = "", lat: float | None = None, lon: float | None = None, radius_km: float = 50, exif: bool = False):
    require_owner(request)
    if len(q) > 250 or radius_km <= 0 or radius_km > 20000:
        raise HTTPException(400, "Invalid search parameters")
    if (lat is None) != (lon is None):
        raise HTTPException(400, "Both GPS coordinates required")
    s3 = client()
    results = []
    scanned = 0
    truncated = False
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket()):
        for obj in page.get("Contents", []):
            scanned += 1
            if scanned > 10000:
                truncated = True
                break
            key = obj["Key"]
            if key.endswith("/") or q.lower() not in key.lower():
                continue
            name = key.rsplit("/", 1)[-1]
            ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
            category = ("image" if ext in ("jpg","jpeg","png","gif","heic","webp","tif","tiff") else
                        "video" if ext in ("mp4","mov","webm","mkv") else
                        "document" if ext in ("pdf","txt","doc","docx","xls","xlsx","csv","ppt","pptx") else "other")
            modified = obj["LastModified"].isoformat()
            if kind and kind != category:
                continue
            if start and modified[:10] < start or end and modified[:10] > end:
                continue
            entry = {"path":key,"name":name,"folder":False,"size":obj["Size"],
                     "modified":modified,"category":category}
            if (exif or lat is not None) and category == "image" and obj["Size"] <= 30000000:
                entry.update(read_exif(s3, key))
            if lat is not None:
                if "latitude" not in entry:
                    continue
                a, b = math.radians(lat), math.radians(entry["latitude"])
                h = math.sin((b-a)/2)**2 + math.cos(a)*math.cos(b)*math.sin(math.radians(entry["longitude"]-lon)/2)**2
                distance = 12742.0176 * math.asin(min(1, math.sqrt(max(0,h))))
                if distance > radius_km:
                    continue
                entry["distance_km"] = round(distance, 2)
            results.append(entry)
            if len(results) >= 500:
                truncated = True
                break
        if truncated:
            break
    return {"items":results,"scanned":min(scanned,10000),"truncated":truncated}


@router.get("/api/preview-url")
def preview_url(request: Request, path: str):
    require_owner(request)
    key = clean_path(path)
    if not key or not key.lower().endswith((".jpg",".jpeg",".png",".webp",".gif",".pdf")):
        raise HTTPException(400, "Preview unavailable for this file type")
    url = client().generate_presigned_url("get_object",
        Params={"Bucket":bucket(),"Key":key}, ExpiresIn=120)
    return {"url":url}

def read_exif(s3, key):
    try:
        data = s3.get_object(Bucket=bucket(), Key=key)["Body"].read(30000001)
        with Image.open(BytesIO(data)) as image:
            ex = image.getexif()
            result = {}
            if ex.get(36867) or ex.get(306):
                result["taken_at"] = str(ex.get(36867) or ex.get(306))
            gps = ex.get_ifd(34853)
            if gps:
                def degrees(values, ref):
                    n = [float(v) for v in values]
                    value = n[0] + n[1]/60 + n[2]/3600
                    return -value if str(ref).upper() in ("S","W") else value
                latitude = degrees(gps[2], gps.get(1))
                longitude = degrees(gps[4], gps.get(3))
                if math.isfinite(latitude) and math.isfinite(longitude):
                    result["latitude"] = round(latitude, 6)
                    result["longitude"] = round(longitude, 6)
            return result
    except (KeyError, ValueError, TypeError, OSError, ZeroDivisionError, IndexError):
        return {}


@router.get("/manifest.webmanifest")
def pwa_manifest():
    return JSONResponse({
        "name": "Alcuin Drive",
        "short_name": "Drive",
        "description": "Private personal cloud storage",
        "start_url": "/storage",
        "scope": "/storage",
        "share_target": {"action": "/storage/share", "method": "POST", "enctype": "multipart/form-data", "params": {"title": "title", "text": "text", "url": "url", "files": [{"name": "files", "accept": ["image/*", "video/*", "application/pdf", "text/plain", ".heic", ".heif"]}]}},
        "display": "standalone",
        "background_color": "#111318",
        "theme_color": "#191b22",
        "shortcuts": [{"name": "Check for updates", "short_name": "Update", "description": "Refresh Alcuin Drive to the latest version", "url": "/storage?check_update=1", "icons": [{"src": "/storage/icon/192.png", "sizes": "192x192"}]}],
        "icons": [
            {"src": "/storage/icon/192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
            {"src": "/storage/icon/512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"}
        ]
    }, media_type="application/manifest+json", headers={"Cache-Control": "public, max-age=3600"})


@router.get("/icon/{size}.png")
def pwa_icon(size: int):
    if size not in (192, 512):
        raise HTTPException(404, "Icon not found")
    from PIL import ImageDraw, ImageFont
    img = Image.new("RGB", (size, size), "#2463ce")
    draw = ImageDraw.Draw(img)
    scale = size / 512
    # Distinctive cloud-and-folder mark with generous maskable safe area.
    draw.rounded_rectangle((int(105*scale), int(195*scale), int(407*scale), int(340*scale)),
                           radius=int(30*scale), fill="#ffffff")
    draw.rounded_rectangle((int(105*scale), int(164*scale), int(257*scale), int(241*scale)),
                           radius=int(21*scale), fill="#ffffff")
    draw.rounded_rectangle((int(142*scale), int(240*scale), int(370*scale), int(262*scale)),
                           radius=int(10*scale), fill="#2463ce")
    stream = BytesIO()
    img.save(stream, format="PNG")
    return Response(stream.getvalue(), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


@router.get("/sw.js")
def pwa_service_worker():
    # Network-only service worker: never cache private files, API responses or sessions.
    script = """self.addEventListener('install', event => self.skipWaiting());
self.addEventListener('activate', event => event.waitUntil(self.clients.claim()));
self.addEventListener('fetch', event => {});
"""
    return Response(script, media_type="application/javascript",
                    headers={"Cache-Control": "no-store", "Service-Worker-Allowed": "/storage/"})


# Incoming Android Web Share Target uploads are staged privately until the user chooses a folder.
SHARE_ROOT = Path("/data/alcuin-share-staging") if Path("/data").is_dir() else Path("/tmp/alcuin-share-staging")
SHARE_MAX_TOTAL = 200 * 1024**2
SHARE_TTL = 3600


def share_dir(token):
    if len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
        raise HTTPException(400, "Invalid share token")
    return SHARE_ROOT / token


def cleanup_shares():
    if not SHARE_ROOT.exists():
        return
    now = time.time()
    for entry in SHARE_ROOT.iterdir():
        if entry.is_dir() and now - entry.stat().st_mtime > SHARE_TTL:
            shutil.rmtree(entry, ignore_errors=True)


@router.post("/share")
async def receive_share(request: Request, files: List[UploadFile] = File(default=[]),
                        title: str = Form(default=""), text: str = Form(default=""),
                        url: str = Form(default="")):
    # Browser share targets submit a form navigation, not JSON. Authentication is still mandatory.
    if not authorized(request) or not configured():
        return Response("<h2>Sign in to Alcuin Drive first</h2><p>Open the installed app, sign in, then share again from Gallery.</p><a href='/storage'>Open Drive</a>",
                        media_type="text/html", status_code=401)
    # Android's OS share sheet can submit Origin: null; browser share targets
    # navigate to this endpoint rather than issuing a same-origin fetch.
    # Require the authenticated, SameSite cookie instead of rejecting that origin.
    cleanup_shares()
    if not files:
        return RedirectResponse("/storage?share_error=no_files", status_code=303)
    token = uuid.uuid4().hex
    directory = share_dir(token)
    directory.mkdir(parents=True, exist_ok=False)
    records = []
    total = 0
    try:
        for index, incoming in enumerate(files):
            filename = Path((incoming.filename or f"shared-{index+1}").replace("\\\\", "/")).name
            if not filename or filename in (".", ".."):
                filename = f"shared-{index+1}"
            path = directory / str(index)
            with path.open("wb") as out:
                while True:
                    chunk = await incoming.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > SHARE_MAX_TOTAL:
                        raise HTTPException(413, "Shared files exceed 200 MB")
                    out.write(chunk)
            records.append({"index": index, "name": filename, "size": path.stat().st_size,
                            "content_type": incoming.content_type or "application/octet-stream"})
        (directory / "meta.json").write_text(json.dumps(records))
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return RedirectResponse("/storage?shared=" + token, status_code=303)


@router.get("/api/shared")
def shared_pending(request: Request, token: str):
    require_owner(request)
    directory = share_dir(token)
    if not directory.is_dir() or time.time() - directory.stat().st_mtime > SHARE_TTL:
        raise HTTPException(404, "Shared files expired; share again from Gallery")
    return {"items": json.loads((directory / "meta.json").read_text())}


class ShareImport(BaseModel):
    token: str
    folder: str = ""


@router.post("/api/import-shared")
def import_shared(request: Request, payload: ShareImport):
    require_owner(request)
    check_origin(request)
    directory = share_dir(payload.token)
    if not directory.is_dir() or time.time() - directory.stat().st_mtime > SHARE_TTL:
        raise HTTPException(404, "Shared files expired; share again")
    prefix = clean_path(payload.folder)
    records = json.loads((directory / "meta.json").read_text())
    s3 = client()
    uploaded = []
    for item in records:
        key = (prefix + "/" if prefix else "") + clean_path(item["name"])
        s3.upload_file(str(directory / str(item["index"])), bucket(), key,
                       ExtraArgs={"ContentType": item["content_type"]})
        uploaded.append(key)
    shutil.rmtree(directory, ignore_errors=True)
    return {"ok": True, "uploaded": uploaded}
