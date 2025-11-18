"""Command-line interface for camera identification (skeleton).

This module will expose a CLI entry point for running the camera
identification pipeline on one or more input images.
"""

import argparse


def main() -> None:
    r"""Entry point for the camera identification CLI.

    For now this only parses basic arguments and reports that the
    implementation is not yet available.
    """
    parser = argparse.ArgumentParser(
        description="Camera Identification & Retrieval CLI (not implemented)",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=False,
        help="Path to input image (not used yet)",
    )

    _ = parser.parse_args()

    print("Camera identification pipeline is not implemented yet.")


if __name__ == "__main__":
    main()
