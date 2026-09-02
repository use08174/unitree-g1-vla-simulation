"""Download GR00T N1.7 to scratch and disable Flash Attention for Tesla T4."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main(args):
    output_directory = Path(args.output_dir)
    snapshot_download(repo_id=args.model, local_dir=output_directory)

    config_path = output_directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["use_flash_attention"] = False
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"GR00T model prepared at {output_directory}")
    print("Flash Attention disabled for Tesla T4 compatibility.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nvidia/GR00T-N1.7-3B")
    parser.add_argument("--output-dir", default="/scratch/sumins/models/GR00T-N1.7-3B")
    main(parser.parse_args())