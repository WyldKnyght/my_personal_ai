# /src/model_handlers/model_download.py

import importlib
from pathlib import Path

def download_model(repo_id, specific_file):
    try:
        print("Downloading model...")
        model_downloader = importlib.import_module("download-model").ModelDownloader()
        model, branch = model_downloader.sanitize_model_and_branch_names(repo_id, None)
        
        links, sha256, is_llamacpp = model_downloader.get_download_links_from_huggingface(model, branch, text_only=False, specific_file=specific_file)
        
        print("Download links:")
        for link in links:
            print(f"`{Path(link).name}`")
        
        output_folder = model_downloader.get_output_folder(model, branch, is_llamacpp=is_llamacpp)
        
        print(f"Downloading file{'s' if len(links) > 1 else ''} to `{output_folder}/`")
        model_downloader.download_model_files(model, branch, links, sha256, output_folder, threads=4, is_llamacpp=is_llamacpp)
        
        print("Download complete!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


