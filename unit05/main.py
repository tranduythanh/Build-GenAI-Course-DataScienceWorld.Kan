import kagglehub as kh

# Download latest version
path = kh.dataset_download("bittlingmayer/amazonreviews")

print("Path to dataset files:", path)