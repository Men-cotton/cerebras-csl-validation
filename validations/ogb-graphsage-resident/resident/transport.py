"""Collect remote raw captures through bounded files, retaining a SHA-bound complete archive."""
import hashlib
import json
from pathlib import Path
import tempfile

PART_BYTES=4<<20


def digest(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def parts(archive,destination):
    record={"archive":"capture.zip","size":archive.stat().st_size,"sha256":digest(archive),"part_bytes":PART_BYTES,"parts":[]}
    with archive.open("rb") as src:
        while chunk:=src.read(PART_BYTES):
            name=f'capture-part-{len(record["parts"]):04d}.bin'
            target=destination/name;target.write_bytes(chunk)
            record["parts"].append({"name":name,"size":len(chunk),"sha256":hashlib.sha256(chunk).hexdigest()})
    (destination/"capture-index.json").write_text(json.dumps(record,indent=2)+"\n")
    return record


def fetch(launcher,output):
    manifest=output/"capture-index.json"
    launcher.download_artifact(manifest.name,str(manifest))
    index=json.loads(manifest.read_text())
    chunks=index["parts"];total=index["size"]
    if not (index["archive"]=="capture.zip" and index["part_bytes"]==PART_BYTES and 0<total<768<<20 and
            len(chunks)==(total+PART_BYTES-1)//PART_BYTES):
        raise ValueError("invalid return file layout")
    archive=output/"capture.zip"
    with tempfile.TemporaryDirectory() as tmp, archive.open("wb") as dest:
        for i,item in enumerate(chunks):
            name=f"capture-part-{i:04d}.bin"
            if item["name"]!=name or item["size"]!=min(PART_BYTES,total-i*PART_BYTES):raise ValueError("invalid return part")
            local=Path(tmp)/name
            launcher.download_artifact(name,str(local))
            if local.stat().st_size!=item["size"] or digest(local)!=item["sha256"]:raise ValueError("changed return part")
            dest.write(local.read_bytes())
            local.unlink()
    if archive.stat().st_size!=total or digest(archive)!=index["sha256"]:raise ValueError("reassembled capture SHA")
    return {"sha256":index["sha256"],"bytes":total,"api_parts":len(chunks),"index_sha256":digest(manifest)}
