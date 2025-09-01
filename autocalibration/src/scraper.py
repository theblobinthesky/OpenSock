import argparse
import csv
import json
import os
import shutil
import pathlib
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from urllib.parse import urlencode

import requests
from tqdm import tqdm

# Constants
FLOOR_KEYWORDS = [
    "floor", "woodfloor", "tile", "tiles", "parquet", "laminate",
    "vinyl", "lvt", "linoleum", "concrete", "terrazzo", "stone",
    "marble", "granite", "carpet", "ground"
]

WANTED_MAPS = [
    "color", "albedo", "basecolor", "base_color",
    "roughness", "normal", "height", "displacement",
    "ao", "ambientocclusion", "ambient_occlusion"
]

DEFAULT_USER_AGENT = "OpenSock-FloorGrabber/1.0 (contact: erik.stern@outlook.com)"

# Central resolution configuration - easily changeable
RESOLUTION_CONFIGS = {
    "1K": {
        "ambientcg": ("1K", "2K", "4K", "8K"),
        "polyhaven": "1k"
    },
    "2K": {
        "ambientcg": ("2K", "1K", "4K", "8K"),
        "polyhaven": "2k"
    },
    "4K": {
        "ambientcg": ("4K", "2K", "1K", "8K"),
        "polyhaven": "4k"
    },
    "8K": {
        "ambientcg": ("8K", "4K", "2K", "1K"),
        "polyhaven": "8k"
    }
}

DEFAULT_SIZE_PREFERENCE = RESOLUTION_CONFIGS["2K"]["ambientcg"]  # Default to 2K
DEFAULT_POLYHAVEN_CATEGORIES = ("floor",)

def ensure_dir(p): pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def download(url, out_path, headers=None, retries=3, sleep=1.5):
    if os.path.exists(out_path): return "skip"
    for i in range(retries):
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers) as r:
                r.raise_for_status()
                ensure_dir(os.path.dirname(out_path))
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(1024 * 128):
                        if chunk: f.write(chunk)
            return "ok"
        except Exception as e:
            if i + 1 == retries: raise
            time.sleep(sleep * (i + 1))
    return "fail"

def guess_map_from_name(name_lower: str):
    # map canonicalization
    if "normal" in name_lower: return "normal"
    if "rough" in name_lower: return "roughness"
    if "disp" in name_lower or "height" in name_lower: return "displacement"
    if "ao" in name_lower or "ambientocclusion" in name_lower: return "ao"
    if "albedo" in name_lower or "basecolor" in name_lower or "_col" in name_lower or "color" in name_lower:
        return "basecolor"
    return None

def guess_res_from_name(name: str):
    # Match patterns like _8K., _8K_, 8K-PNG, 4K-JPG, or in filenames
    m = re.search(r"([0-9]{1,2})K(?=[_\.-])", str(name).upper())
    return f"{m.group(1)}K" if m else None

def get_floor_categories():
    """Get list of floor-related categories from AmbientCG."""
    categories_url = "https://ambientcg.com/api/v2/categories_json"
    res = requests.get(categories_url, timeout=30)
    res.raise_for_status()
    data = res.json()

    floor_keywords = [
        "floor", "woodfloor", "tile", "tiles", "parquet", "laminate",
        "vinyl", "lvt", "linoleum", "concrete", "terrazzo", "stone",
        "marble", "granite", "carpet", "ground", "paving"
    ]

    floor_cats = []
    for item in data:
        cat_name = item["categoryName"].lower()
        if any(kw in cat_name for kw in floor_keywords):
            floor_cats.append(item["categoryName"])

    return floor_cats


def fetch_ambientcg_assets(headers, max_assets):
    """Fetch AmbientCG assets from floor-related categories using CSV endpoint."""
    base_url = "https://ambientcg.com/api/v2/downloads_csv"

    # Get floor-related categories
    floor_categories = get_floor_categories()
    print(f"Found {len(floor_categories)} floor-related categories: {', '.join(floor_categories[:5])}{'...' if len(floor_categories) > 5 else ''}")

    all_downloads = []

    # Fetch downloads from each category
    for category in floor_categories:
        params = {
            "type": "Material",
            "category": category,
            "limit": 250  # Max allowed by API
        }

        try:
            res = requests.get(base_url, params=params, headers=headers, timeout=60)
            res.raise_for_status()

            # Parse CSV
            csv_content = res.text
            lines = csv_content.strip().split('\n')
            if len(lines) < 2:  # No data rows
                continue

            # Parse header
            reader = csv.DictReader(StringIO(csv_content))
            for row in reader:
                all_downloads.append({
                    "asset_id": row["assetId"],
                    "attribute": row["downloadAttribute"],
                    "filetype": row["filetype"],
                    "size": int(row["size"]),
                    "downloadLink": row["downloadLink"],
                    "rawLink": row["rawLink"],
                    "category": category
                })

            # Break if we have enough downloads
            if max_assets and len(all_downloads) >= max_assets:
                break

        except Exception as e:
            print(f"Error fetching category {category}: {e}")
            continue

    # Group downloads by asset and select best resolution
    asset_downloads = {}
    for download in all_downloads:
        asset_id = download["asset_id"]
        if asset_id not in asset_downloads:
            asset_downloads[asset_id] = []
        asset_downloads[asset_id].append(download)

    # Limit results
    if max_assets:
        asset_ids = list(asset_downloads.keys())[:max_assets]
        asset_downloads = {aid: asset_downloads[aid] for aid in asset_ids}

    return asset_downloads

    res = requests.get(base_url, params=params, headers=headers, timeout=60)
    res.raise_for_status()
    data = res.json()

    if not data or "data" not in data:
        return {}

    assets = data["data"]
    return dict(list(assets.items())[:max_assets]) if max_assets else assets


def select_preferred_size(available_sizes, size_preference):
    """Select the best available size from preferences."""
    for pref in size_preference:
        if pref in available_sizes:
            return pref
    return available_sizes[0] if available_sizes else None


def download_ambientcg_asset(asset_id, downloads, size_preference, out_root, headers):
    """Download all maps for a single AmbientCG asset."""
    if not downloads:
        tqdm.write(f"[AmbientCG] {asset_id}: no downloads available")
        return None

    # Find preferred resolution ZIP
    selected_download = None
    selected_size = None

    for download_info in downloads:
        attribute = download_info["attribute"]
        size_match = re.match(r"(\d+)K", attribute)
        if size_match:
            size = f"{size_match.group(1)}K"
            if size in size_preference:
                selected_download = download_info
                selected_size = size
                break

    # If no preferred size found, take the largest available
    if not selected_download:
        # Sort by size descending
        sorted_downloads = sorted(downloads, key=lambda x: x["size"], reverse=True)
        selected_download = sorted_downloads[0]
        attribute = selected_download["attribute"]
        size_match = re.match(r"(\d+)K", attribute)
        selected_size = f"{size_match.group(1)}K" if size_match else "Unknown"

    if not selected_size:
        print(f"[AmbientCG] {asset_id}: could not determine size")
        return None

    base_path = os.path.join(out_root, "AmbientCG", asset_id, selected_size.upper())
    ensure_dir(base_path)

    meta_out = {
        "asset_id": asset_id,
        "provider": "AmbientCG",
        "size": selected_size,
        "name": asset_id,  # We don't have display name from CSV
        "categories": [downloads[0]["category"]],  # Use category from first download
        "files": {}
    }

    # Download and extract ZIP
    zip_url = selected_download["downloadLink"]
    zip_filename = f"{asset_id}_{selected_size}.zip"
    zip_path = os.path.join(base_path, zip_filename)

    tqdm.write(f"[AmbientCG] {asset_id}: downloading {selected_size} ZIP...")
    status = download(zip_url, zip_path, headers=headers)
    if status not in ["ok", "skip"]:
        tqdm.write(f"[AmbientCG] {asset_id}: failed to download ZIP")
        return None

    # Extract ZIP
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            extracted_files = zip_ref.namelist()
            zip_ref.extractall(base_path)
        tqdm.write(f"[AmbientCG] {asset_id}: extracted ZIP with {len(extracted_files)} files")
    except Exception as e:
        tqdm.write(f"[AmbientCG] {asset_id}: failed to extract ZIP: {e}")
        return None

    # Map extracted files to canonical types
    file_mappings = {}

    for filename in extracted_files:
        if filename.endswith(('.jpg', '.png')) and not filename.startswith('__MACOSX'):
            # Parse filename to determine map type
            if '_Color.' in filename or '_Albedo.' in filename:
                file_mappings["basecolor"] = filename
            elif '_Roughness.' in filename:
                file_mappings["roughness"] = filename
            elif '_NormalGL.' in filename or '_Normal.' in filename:
                file_mappings["normal"] = filename
            elif '_Displacement.' in filename or '_Height.' in filename:
                file_mappings["displacement"] = filename
            elif '_AmbientOcclusion.' in filename or '_AO.' in filename:
                file_mappings["ao"] = filename

    # Copy files to canonical names
    for canon_type, filename in file_mappings.items():
        src_path = os.path.join(base_path, filename)
        ext = os.path.splitext(filename)[1]
        dst_path = os.path.join(base_path, f"{canon_type}{ext}")

        if os.path.exists(src_path):
            # Copy file to canonical name
            with open(src_path, 'rb') as src, open(dst_path, 'wb') as dst:
                dst.write(src.read())

            meta_out["files"][canon_type] = {
                "url": zip_url,
                "file": os.path.relpath(dst_path, os.path.join(out_root, "AmbientCG", asset_id))
            }
            tqdm.write(f"[AmbientCG] {asset_id} {canon_type}@{selected_size}: extracted")
        else:
            tqdm.write(f"[AmbientCG] {asset_id} {canon_type}@{selected_size}: file not found")

    # Save metadata
    with open(os.path.join(os.path.dirname(base_path), "meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    return meta_out


def ambientcg_download(out_root, size_preference=DEFAULT_SIZE_PREFERENCE, max_per_asset=None, keywords=FLOOR_KEYWORDS):
    """Download floor-like PBR textures from AmbientCG using their API v2."""
    headers = {"User-Agent": "OpenSock-FloorGrabber/1.0 (contact: erik.stern@outlook.com)"}
    total_files_downloaded = 0

    try:
        assets = fetch_ambientcg_assets(headers, max_per_asset)
        tqdm.write(f"AmbientCG: {len(assets)} assets found")

        # Parallel download with progress bar
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit download tasks one by one, checking file limit
            future_to_asset = {}
            assets_list = list(assets.items())

            with tqdm(total=len(assets_list), desc="Downloading AmbientCG") as pbar:
                for asset_id, downloads in assets_list:
                    # Submit download task
                    future = executor.submit(download_ambientcg_asset, asset_id, downloads, size_preference, out_root, headers)
                    future_to_asset[future] = asset_id

                    # Process completed downloads and check limits
                    completed_futures = [f for f in future_to_asset.keys() if f.done()]
                    for future in completed_futures:
                        asset_id = future_to_asset[future]
                        try:
                            result = future.result()
                            if result and isinstance(result, dict) and result.get("files"):
                                files_count = len(result["files"])
                                total_files_downloaded += files_count

                            if not result:
                                tqdm.write(f"[AmbientCG] {asset_id}: skipped (no suitable download found)")
                        except Exception as e:
                            tqdm.write(f"Error processing AmbientCG asset {asset_id}: {e}")

                        del future_to_asset[future]
                        pbar.update(1)

                # Process any remaining completed futures (but don't add to count if over limit)
                for future in as_completed(future_to_asset.keys()):
                    asset_id = future_to_asset[future]
                    try:
                        result = future.result()
                        if result and isinstance(result, dict) and result.get("files"):
                            files_count = len(result["files"])
                            total_files_downloaded += files_count

                        if not result:
                            tqdm.write(f"[AmbientCG] {asset_id}: skipped (no suitable download found)")
                    except Exception as e:
                        tqdm.write(f"Error processing AmbientCG asset {asset_id}: {e}")
                    pbar.update(1)

    except Exception as e:
        tqdm.write(f"AmbientCG API error: {e}")

    return total_files_downloaded


def fetch_polyhaven_assets(headers, category_filter, max_assets):
    """Fetch and filter PolyHaven assets by categories."""
    assets_url = "https://api.polyhaven.com/assets?t=textures"
    res = requests.get(assets_url, timeout=60, headers=headers)
    res.raise_for_status()
    data = res.json()

    items = []
    if isinstance(data, dict):
        for aid, meta in data.items():
            cats = [c.lower() for c in (meta.get("categories") or [])]
            if any(cf in cats for cf in category_filter):
                items.append((aid, meta))
    elif isinstance(data, list):
        for meta in data:
            aid = meta.get("id") or meta.get("name")
            cats = [c.lower() for c in (meta.get("categories") or [])]
            if aid and any(cf in cats for cf in category_filter):
                items.append((aid, meta))

    return items[:max_assets] if max_assets else items


def fetch_polyhaven_hdri_assets(headers, category_filter, max_assets):
    """Fetch and filter PolyHaven HDRI assets by categories, focusing on indoor lighting."""
    assets_url = "https://api.polyhaven.com/assets?t=hdris"
    res = requests.get(assets_url, timeout=60, headers=headers)
    res.raise_for_status()
    data = res.json()

    items = []
    if isinstance(data, dict):
        for aid, meta in data.items():
            cats = [c.lower() for c in (meta.get("categories") or [])]
            # Filter for indoor lighting categories
            if any(cf in cats for cf in category_filter):
                items.append((aid, meta))
    elif isinstance(data, list):
        for meta in data:
            aid = meta.get("id") or meta.get("name")
            cats = [c.lower() for c in (meta.get("categories") or [])]
            if aid and any(cf in cats for cf in category_filter):
                items.append((aid, meta))

    return items[:max_assets] if max_assets else items


def download_polyhaven_asset(aid, size, out_root, headers):
    """Download all maps for a single PolyHaven asset."""
    try:
        files_url = f"https://api.polyhaven.com/files/{aid}"
        fjs = requests.get(files_url, timeout=60, headers=headers).json()

        # PolyHaven structure: {map_type: {size: url}}
        # Map PolyHaven types to canonical types
        type_mapping = {
            "Diffuse": "basecolor",
            "Rough": "roughness",
            "nor_gl": "normal",
            "Displacement": "displacement",
            "AO": "ao"
        }

        size_key = size.lower()

        base = os.path.join(out_root, "PolyHaven", aid, size_key.upper())
        ensure_dir(base)

        meta_out = {
            "asset_id": aid,
            "provider": "PolyHaven",
            "size": size_key,
            "files": {}
        }

        downloaded_any = False
        for ph_type, canon_type in type_mapping.items():
            if ph_type in fjs and size_key in fjs[ph_type]:
                size_data = fjs[ph_type][size_key]

                # Choose format: PNG if available, otherwise JPG
                if isinstance(size_data, dict):
                    if "png" in size_data:
                        format_data = size_data["png"]
                        ext = ".png"
                    elif "jpg" in size_data:
                        format_data = size_data["jpg"]
                        ext = ".jpg"
                    else:
                        continue

                    url = format_data["url"]
                else:
                    # Fallback for direct URL
                    url = size_data
                    ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"

                out_path = os.path.join(base, f"{canon_type}{ext}")
                status = download(url, out_path, headers=headers)
                if status in ["ok", "skip"]:
                    tqdm.write(f"[PolyHaven] {aid} {canon_type}@{size_key}: {status}")
                    meta_out["files"][canon_type] = {
                        "url": url,
                        "file": os.path.relpath(out_path, os.path.join(out_root, "PolyHaven", aid))
                    }
                    downloaded_any = True
            else:
                tqdm.write(f"[PolyHaven] {aid} {canon_type}@{size_key}: not available")

        if downloaded_any:
            # Save metadata
            with open(os.path.join(os.path.dirname(base), "meta.json"), "w") as f:
                json.dump(meta_out, f, indent=2)
            return len(meta_out["files"])  # Return number of files downloaded
        else:
            tqdm.write(f"[PolyHaven] {aid}: no maps downloaded")
            return 0

    except Exception as e:
        tqdm.write(f"[PolyHaven] Error downloading {aid}: {e}")
        return 0


def download_polyhaven_hdri_asset(aid, size, out_root, headers, hdri_counter):
    """Download HDRI files for a single PolyHaven HDRI asset."""
    try:
        files_url = f"https://api.polyhaven.com/files/{aid}"
        fjs = requests.get(files_url, timeout=60, headers=headers).json()

        size_key = size.lower()

        # Create flat HDRIs directory
        hdri_dir = os.path.join(out_root, "hdris")
        ensure_dir(hdri_dir)

        # Download main HDRI file
        if "hdri" in fjs and size_key in fjs["hdri"]:
            hdri_data = fjs["hdri"][size_key]

            # Choose format: HDR if available, otherwise EXR
            if isinstance(hdri_data, dict):
                if "hdr" in hdri_data:
                    format_data = hdri_data["hdr"]
                    ext = ".hdr"
                elif "exr" in hdri_data:
                    format_data = hdri_data["exr"]
                    ext = ".exr"
                else:
                    tqdm.write(f"[PolyHaven-HDRI] {aid}: no suitable HDRI format found")
                    return 0

                url = format_data["url"]
            else:
                # Fallback for direct URL
                url = hdri_data
                ext = os.path.splitext(url.split("?")[0])[1] or ".hdr"

            # Use sequential naming
            out_path = os.path.join(hdri_dir, f"{hdri_counter:04d}{ext}")
            status = download(url, out_path, headers=headers)
            if status in ["ok", "skip"]:
                tqdm.write(f"[PolyHaven-HDRI] {aid} -> {hdri_counter:04d}{ext} @{size_key}: {status}")
                return 1  # Return 1 for successful download
            else:
                tqdm.write(f"[PolyHaven-HDRI] {aid}: failed to download HDRI")
                return 0
        else:
            tqdm.write(f"[PolyHaven-HDRI] {aid}: HDRI file not available at {size_key}")
            return 0

    except Exception as e:
        tqdm.write(f"[PolyHaven-HDRI] Error downloading {aid}: {e}")
        return 0


def polyhaven_download(out_root, size="8k", category_filter=DEFAULT_POLYHAVEN_CATEGORIES, max_assets=None, user_agent=DEFAULT_USER_AGENT):
    """Download floor-like PBR textures from PolyHaven."""
    headers = {"User-Agent": user_agent}
    total_files_downloaded = 0

    items = fetch_polyhaven_assets(headers, category_filter, max_assets)
    tqdm.write(f"PolyHaven: {len(items)} assets")

    # Parallel download with progress bar
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit download tasks one by one, checking file limit
        future_to_asset = {}
        items_list = list(items)

        with tqdm(total=len(items_list), desc="Downloading PolyHaven") as pbar:
            for aid, meta in items_list:
                # Submit download task
                future = executor.submit(download_polyhaven_asset, aid, size, out_root, headers)
                future_to_asset[future] = aid

                # Process completed downloads and check limits
                completed_futures = [f for f in future_to_asset.keys() if f.done()]
                for future in completed_futures:
                    aid = future_to_asset[future]
                    try:
                        files_count = future.result()
                        total_files_downloaded += files_count
                    except Exception as e:
                        tqdm.write(f"Error downloading PolyHaven asset {aid}: {e}")

                    del future_to_asset[future]
                    pbar.update(1)

            # Process any remaining completed futures (but don't add to count if over limit)
            for future in as_completed(future_to_asset.keys()):
                aid = future_to_asset[future]
                try:
                    files_count = future.result()
                    total_files_downloaded += files_count
                except Exception as e:
                    tqdm.write(f"Error downloading PolyHaven asset {aid}: {e}")
                pbar.update(1)

    return total_files_downloaded


def polyhaven_hdri_download(out_root, size="8k", category_filter=("indoor", "artificial light"), max_assets=None, user_agent=DEFAULT_USER_AGENT):
    """Download indoor lighting HDRI files from PolyHaven."""
    headers = {"User-Agent": user_agent}
    total_files_downloaded = 0
    hdri_counter = 0  # Sequential counter for naming

    items = fetch_polyhaven_hdri_assets(headers, category_filter, max_assets)
    tqdm.write(f"PolyHaven-HDRIs: {len(items)} indoor lighting HDRIs found")

    # Parallel download with progress bar
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit download tasks one by one, checking file limit
        future_to_asset = {}
        items_list = list(items)

        with tqdm(total=len(items_list), desc="Downloading PolyHaven HDRIs") as pbar:
            for aid, meta in items_list:
                # Submit download task with current counter
                future = executor.submit(download_polyhaven_hdri_asset, aid, size, out_root, headers, hdri_counter)
                future_to_asset[future] = (aid, hdri_counter)
                hdri_counter += 1  # Increment counter for next asset

                # Process completed downloads and check limits
                completed_futures = [f for f in future_to_asset.keys() if f.done()]
                for future in completed_futures:
                    aid, counter = future_to_asset[future]
                    try:
                        success = future.result()
                        if success:
                            total_files_downloaded += 1
                        else:
                            tqdm.write(f"[PolyHaven-HDRI] Failed to download {aid}")
                    except Exception as e:
                        tqdm.write(f"Error downloading PolyHaven HDRI asset {aid}: {e}")

                    del future_to_asset[future]
                    pbar.update(1)

            # Process any remaining completed futures
            for future in as_completed(future_to_asset.keys()):
                aid, counter = future_to_asset[future]
                try:
                    success = future.result()
                    if success:
                        total_files_downloaded += 1
                    else:
                        tqdm.write(f"[PolyHaven-HDRI] Failed to download {aid}")
                except Exception as e:
                    tqdm.write(f"Error downloading PolyHaven HDRI asset {aid}: {e}")
                pbar.update(1)

    return total_files_downloaded


def run_scraper(out="data/scraped-assets", provider="both", max_materials: int=1, max_hdris: int=1, size="1K"):
    shutil.rmtree(out, ignore_errors=True)

    # Get resolution configuration
    if size.upper() in RESOLUTION_CONFIGS:
        config = RESOLUTION_CONFIGS[size.upper()]
        ambientcg_sizes = config["ambientcg"]
        polyhaven_size = config["polyhaven"]
    else:
        # Fallback to default
        ambientcg_sizes = DEFAULT_SIZE_PREFERENCE
        polyhaven_size = size.lower()

    total_files_downloaded = 0

    if provider in ("ambientcg", "both"):
        tqdm.write(f"Starting AmbientCG download (size preference: {size})...")
        files_downloaded = ambientcg_download(
            out,
            size_preference=ambientcg_sizes,
            max_per_asset=max_materials,
            keywords=FLOOR_KEYWORDS
        )
        total_files_downloaded += files_downloaded or 0

    if provider in ("polyhaven", "both"):
        tqdm.write(f"Starting PolyHaven download (size: {size})...")
        files_downloaded = polyhaven_download(
            out,
            size=polyhaven_size,
            category_filter=DEFAULT_POLYHAVEN_CATEGORIES,
            max_assets=max_materials,
            user_agent=DEFAULT_USER_AGENT
        )
        total_files_downloaded += files_downloaded or 0

    if provider in ("hdris", "both"):
        tqdm.write(f"Starting PolyHaven HDRI download (size: {size})...")
        files_downloaded = polyhaven_hdri_download(
            out,
            size=polyhaven_size,
            category_filter=("indoor", "artificial light"),
            max_assets=max_hdris,
            user_agent=DEFAULT_USER_AGENT
        )
        total_files_downloaded += files_downloaded or 0


def main():
    """Main entry point for the texture scraper."""
    parser = argparse.ArgumentParser(
        description="Download floor-like PBR textures and indoor lighting HDRIs from AmbientCG and PolyHaven.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
 Examples:
   python -m src.scraper                           # Download materials from both providers (2K default)
   python -m src.scraper --size 1K --max-files 50  # 1K textures, limit to 50 total files
   python -m src.scraper --provider ambientcg --max 10  # Only AmbientCG materials, limit 10 assets
   python -m src.scraper --provider hdris --max 20      # Download 20 indoor lighting HDRIs
   python -m src.scraper --size 4K --out my_assets/      # 4K assets to custom directory
         """
    )

    parser.add_argument(
        "--out", "-o",
        default="data/scraped-assets",
        help="Output directory for downloaded textures"
    )
    parser.add_argument(
        "--provider", "-p",
        choices=["ambientcg", "polyhaven", "hdris", "both"],
        default="both",
        help="Provider(s) to use: ambientcg (materials), polyhaven (materials), hdris (HDRIs), both (materials from both providers)"
    )
    parser.add_argument(
        "--max", "-m",
        type=int,
        default=100,
        help="Maximum assets to download per provider"
    )
    parser.add_argument(
        "--max-files", "-f",
        type=int,
        default=None,
        help="Maximum total files to download across all providers"
    )
    parser.add_argument(
        "--size", "-s",
        choices=["1K", "2K", "4K", "8K"],
        default="1K",
        help="Preferred texture size (1K, 2K, 4K, or 8K - automatically configured for both providers)"
    )

    args = parser.parse_args()

    tqdm.write(f"Texture Scraper - Provider: {args.provider}, Size: {args.size}")
    if args.max:
        tqdm.write(f"Max assets per provider: {args.max}")
    if args.max_files:
        tqdm.write(f"Max total files: {args.max_files}")

    run_scraper(
        out=args.out,
        provider=args.provider,
        max_assets=args.max,
        max_files=args.max_files,
        size=args.size
    )

    tqdm.write("Scraping completed!")

if __name__ == "__main__":
    main()
