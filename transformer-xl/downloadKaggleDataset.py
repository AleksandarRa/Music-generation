import kagglehub

# classical Music ML Format
link_classical = "jembishop1/classical-music-piano-rolls"
link_jazz = "saikayala/jazz-ml-ready-midi"
# Download latest version
path = kagglehub.dataset_download(link_jazz)

print("Path to dataset files:", path)