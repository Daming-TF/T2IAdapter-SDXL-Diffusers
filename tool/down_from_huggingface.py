vision = 'SargeZT/t2i-adapter-sdxl-multi'
dir= r'/mnt/nfs/file_server/public/mingjiahui/models/SargeZT--t2i-adapter-sdxl-multi'

from huggingface_hub import snapshot_download
snapshot_download(repo_id=vision, local_dir=dir)