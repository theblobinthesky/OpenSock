import argparse
import subprocess
import sys


def run_command(cmd: str, desc: str):
    print(f"Running: {desc}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(f"Error in {desc}: {result.stderr}")
        sys.exit(result.returncode)
    print(f"Completed: {desc}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Autocalibration pipeline")
    parser.add_argument("--scrape", action="store_true", help="Run texture scraping")
    parser.add_argument("--render", action="store_true", help="Run Blender rendering")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip feature extraction stage")
    parser.add_argument("--skip-train", action="store_true", help="Skip training stage")
    parser.add_argument("--inputs", default="data/input", help="Input image directory")
    parser.add_argument("--dino-out", default="data/dino-features", help="Output dir for DINO-like features")
    parser.add_argument("--sp-out", default="data/superpoint-features", help="Output dir for SuperPoint-like features")
    args = parser.parse_args(argv)

    if args.scrape:
        print("Running: Scraping textures")
        from .scraper import run_scraper
        run_scraper(max_assets=5)
        print("Completed: Scraping textures")

    if args.render:
        run_command("blender -b -P src/blender_render.py", "Rendering with Blender")

    if not args.skip_preprocess:
        run_command(
            f"python src/preprocess.py",
            "Preprocessing images",
        )
    else:
        print("Skipping preprocess (per flag)")

    if not args.skip_train:
        run_command(
            f"python src/train.py",
            "Training model",
        )
    else:
        print("Skipping train (per flag)")

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
