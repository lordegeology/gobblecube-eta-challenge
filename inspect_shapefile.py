"""Run this locally to see what's inside the TLC zip.
   python inspect_shapefile.py
"""
import urllib.request, zipfile, io

zip_url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
print("Downloading zip...")
with urllib.request.urlopen(zip_url) as r:
    zdata = r.read()

with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
    print("\nAll files in zip:")
    for name in zf.namelist():
        print(f"  {name}")
