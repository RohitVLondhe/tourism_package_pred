from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

api = HfApi(token=os.getenv("HF_TOKEN"))

repo_id = "Pvt-Pixel/tourism-package-prediction-app"
repo_type = "space"

# Step 1: Check if the space exists, if not create it
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id=repo_id,          # the target repo
    repo_type=repo_type,                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
