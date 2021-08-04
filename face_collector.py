#!/usr/bin/env python
import face_recognition
import cv2
import youtube_dl
import ffmpeg
import argparse
import tempfile
from pathlib import Path
import random
import time
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="url of the video to download and where to detect faces")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tempdir:
        video_path = f"{tempdir}/video.mp4"

        params = {"outtmpl": video_path, "restrictfilenames": True}

        with youtube_dl.YoutubeDL(params) as ydl:
            ydl.download([args.url])

        ffmpeg.input(video_path, ss=random.random()).filter("fps", fps=1).output(
            f"{tempdir}/output_%d.jpg"
        ).run(capture_stdout=True)

        output_dir = str(int(time.time() * 1000))
        os.makedirs(output_dir)

        images = [
            file for file in Path(tempdir).iterdir() if file.is_file() and file.suffix == ".jpg"
        ]

        print("Images extracted:", len(images))

        for i, image_path in enumerate(images):
            image_path = str(image_path)
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)

            print(f"Faces count for image {i}: {len(face_locations)}")

            if not face_locations:
                continue

            image = cv2.imread(image_path)

            for j, (top, right, bottom, left) in enumerate(face_locations):
                extra = int(max(right - left, bottom - top) * 0.2)
                face = image[
                    max(0, top - extra) : bottom + extra, max(0, left - extra) : right + extra
                ]
                cv2.imwrite(f"{output_dir}/face_{i}_{j}.jpg", face)


if __name__ == "__main__":
    main()
