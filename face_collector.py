#!/usr/bin/env python
import face_recognition
import cv2
import youtube_dl
import ffmpeg
import argparse
import tempfile
import sys
from pathlib import Path
import random
import time
import os
import shutil


from loguru import logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "url", nargs="?", help="url of the video to download and where to detect faces"
    )
    parser.add_argument(
        "--clear-cache",
        help="remove cache folder used to store downloaded videos",
        action="store_true",
    )
    parser.add_argument(
        "--verbose",
        help="show all logs",
        action="store_true",
    )
    args = parser.parse_args()

    url = args.url
    clear_cache = args.clear_cache
    verbose = args.verbose

    logger.configure(handlers=[dict(sink=sys.stderr, level="TRACE" if verbose else "INFO")])

    cachedir = Path(tempfile.gettempdir()) / "face_collector_cache"

    if clear_cache:
        logger.debug(f"Removing cache directory: '{cachedir}'")
        shutil.rmtree(str(cachedir))

    if url is None:
        logger.info("No video to process")
        return

    logger.info(f"Preparing download of video '{url}'")

    with youtube_dl.YoutubeDL() as ydl:
        info_dict = ydl.extract_info(url, download=False)

    logger.trace(f"Video metadata: {info_dict}")

    media_id = info_dict["id"]
    video_path = cachedir / f"{media_id}.mp4"

    if video_path.exists():
        logger.info("Loading video file from cache")
    else:
        logger.info("Downloading video file (not cached)")
        with youtube_dl.YoutubeDL({"outtmpl": str(video_path), "restrictfilenames": True}) as ydl:
            ydl.download([url])

    with tempfile.TemporaryDirectory() as tempdir:
        logger.info("Sampling video to extract images.")
        ffmpeg.input(video_path, ss=random.random()).filter("fps", fps=1).output(
            f"{tempdir}/output_%d.jpg"
        ).run(capture_stdout=True)

        output_dir = f"output/{int(time.time() * 1000)}"
        os.makedirs(output_dir, exist_ok=True)

        images = [
            file for file in Path(tempdir).iterdir() if file.is_file() and file.suffix == ".jpg"
        ]

        logger.info(f"Analyzing {len(images)} images extracted.")

        for i, image_path in enumerate(images, start=1):
            image_path = str(image_path)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)

            logger.debug(f"Faces count for image {i}: {len(face_locations)}")

            if not face_locations:
                continue

            image = cv2.imread(image_path)

            for j, (top, right, bottom, left) in enumerate(face_locations):
                extra = int(max(right - left, bottom - top) * 0.2)
                face = image[
                    max(0, top - extra) : bottom + extra, max(0, left - extra) : right + extra
                ]
                cv2.imwrite(f"{output_dir}/face_{i}_{j}.jpg", face)

    logger.info(f"Video processed successfully, faces saved in '{output_dir}'")


if __name__ == "__main__":
    main()
