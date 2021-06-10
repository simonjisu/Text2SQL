import gdown
from pathlib import Path

base_url = "https://drive.google.com/uc?id={}"
urls = ["1FTP6cDCC-hyeXupOmgNw8JN3IRwk3YXG"]
urls = [base_url.format(x) for x in urls]
ckpt_path = Path(".") / "ckpt"
if not ckpt_path.exists():
    output_folder.mkdir()
for i, url in enumerate(urls, 1):
    gdown.download(url, str(output_folder / f"checkpoint{i}.ckpt"), quiet=False)
