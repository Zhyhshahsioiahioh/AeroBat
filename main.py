"""
Entry point for the AeroBat demo.

Usage:
  python main.py                   # Launch PyQt6 UI (default)
  python main.py --cli --config config/tad.yaml --mode human
"""
import argparse
import sys
import time
from pathlib import Path

import imageio.v2 as imageio

# Ensure repository root is on PYTHONPATH so that `src` is importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.scenarios import make_env, load_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run AeroBat environment demo (GUI by default).")
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "tad.yaml"),
        help="Path to YAML config (defaults to config/tad.yaml).",
    )
    parser.add_argument(
        "--mode",
        choices=["human", "rgb_array"],
        default="human",
        help="Render mode for CLI runs.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.05,
        help="Delay between frames when mode=human.",
    )
    parser.add_argument(
        "--cli",
        "--no-gui",
        action="store_true",
        dest="cli",
        help="Run in command-line mode without the PyQt6 UI.",
    )
    return parser.parse_args()


def run_cli(args):
    cfg = load_config(args.config)
    if not hasattr(cfg, "save_gif"):
        cfg.save_gif = False
    if not hasattr(cfg, "gif_dir"):
        cfg.gif_dir = "gifs"
    frames_count = getattr(cfg, "episode_length", 300)

    env = make_env(config=cfg)
    env.reset()

    gif_frames = []
    gif_dir = Path(cfg.gif_dir)
    if not gif_dir.is_absolute():
        gif_dir = ROOT / gif_dir

    try:
        for _ in range(int(frames_count)):
            env.step()
            if cfg.save_gif:
                frame = env.render(mode="rgb_array")[0]
                gif_frames.append(frame)
                if args.mode == "human":
                    env.render(mode="human")
            else:
                env.render(mode=args.mode)
            if args.mode == "human":
                time.sleep(args.sleep)
    finally:
        if cfg.save_gif and len(gif_frames) > 0:
            gif_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            gif_path = gif_dir / f"render_{timestamp}.gif"
            imageio.mimsave(gif_path, gif_frames, duration=args.sleep)
            print(f"Saved GIF to {gif_path}")
        env.close()

import sys
from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

