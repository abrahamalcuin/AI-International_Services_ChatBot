"""Private R2 drive. Configure R2_* and DRIVE_PASSWORD_HASH on Render.

Independent owner-only login; existing employee sessions do not grant access.
"""
import hashlib
import hmac
import os
import secrets
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from io import BytesIO
from PIL import Image
import math

import boto3
from botocore.config import Config
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/storage", tags=["private-drive"])
PAGE = Path(__file__).with_name("storage_ui.html")
MAX_UPLOAD_BYTES = 5 * 1024**3
MAX_PROXY_UPLOAD_BYTES = 200 * 1024**2
_login_attempts = {}
_login_lock = threading.Lock()


def configured():
    return all(os.getenv(k) for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "DRIVE_PASSWORD_HASH"))


def bucket():
    return os.environ["R2_BUCKET"]


def client():
    return boto3.client("s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto", config=Config(signature_version="s3v4"))


def authorized(request):
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
    return {"ok": True}


@router.post("/api/logout")
def logout(request: Request):
    check_origin(request)
    request.session.pop("private_drive_owner", None)
    return {"ok": True}


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
