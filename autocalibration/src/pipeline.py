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
    parser.add_argument("--inputs", default="data/input", help="Input image directory")
    parser.add_argument("--dino-out", default="data/dino-features", help="Output dir for DINO-like features")
    parser.add_argument("--sp-out", default="data/superpoint-features", help="Output dir for SuperPoint-like features")

    max_materials = 300
    max_hdris = 100
    num_scenes = 20000

    args = parser.parse_args(argv)

    print("Running: Scraping textures")
    from scraper import run_scraper
    run_scraper(max_materials=max_materials, max_hdris=max_hdris)
    print("Completed: Scraping textures")

    output_dir = "data/dataset"
    run_command(
        f"blender -b -P src/blender_render.py -- --num-scenes {num_scenes} --output-dir {output_dir} --keep-rigidbody --save-blend",
        "Rendering images"
    )
    exit(0)

    run_command(
        f"python src/preprocess.py",
        "Preprocessing images",
    )

    # run_command(
    #     f"python src/train.py",
    #     "Training model",
    # )

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
