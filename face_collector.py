#!/usr/bin/env python
import face_recognition
import cv2
import youtube_dl
import ffmpeg
import argparse
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="url of the video to download and where to detect faces")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tempdir:
        video_path = f"{tempdir}/test.mp4"

        params = {"outtmpl": video_path, "restrictfilenames": True}

        with youtube_dl.YoutubeDL(params) as ydl:
            ydl.download([args.url])

        stream = ffmpeg.input(video_path)
        out, _ = (
            stream.filter("select", "gte(n,{})".format(1825))
            .output("pipe:", vframes=1, format="image2", vcodec="mjpeg")
            .run(capture_stdout=True)
        )

        image_path = f"{tempdir}/test.jpg"

        with open(image_path, "wb") as file:
            file.write(out)

        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)

        image = cv2.imread(image_path)

        print("Faces count: {}", len(face_locations))

        for i, (top, right, bottom, left) in enumerate(face_locations):
            face = image[top:bottom, left:right]
            cv2.imwrite(f"face_{i}.jpg", face)


if __name__ == "__main__":
    main()
