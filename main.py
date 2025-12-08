from huggingface_hub import snapshot_download


def main():
    model_id="TomoroAI/tomoro-colqwen3-embed-4b"
    snapshot_download(repo_id=model_id, local_dir="tomoro-colqwen3-embed-4b",
      local_dir_use_symlinks=False, revision="main")

if __name__ == "__main__":
    main()
