from __future__ import annotations
from os.path import join

import cv2
import numpy as np

class Recorder():
    """The recorder records a run of a simulation and saves it as a video file
    """

    def __init__(self,
                fps:int=15,
                height:int=480,
                width:int=640,
                folder:str='videos') -> None:
        """Constructor for the recorder
        Args:
            fps (int, optional): Number of fps for video clip. Defaults to 15.
            height (int, optional): Height of video. Defaults to 480.
            width (int, optional): Width of video. Defaults to 640.
            folder (str, optional): Storage location. Defaults to 'videos'.
        """

        self.cameraCapture = cv2.VideoCapture(0)
        self.height = height
        self.width = width
        self.fps = float(fps)
        self.folder = folder

        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video = None

    def init_new_video(self, id:str) -> None:
        """Triggers the process of recording a new video 
        Args:
            id (str): Id/name of the video file
        """
        self.video = cv2.VideoWriter(join(self.folder, f'{id}.mp4'),
                                        self.fourcc,
                                        self.fps,
                                        (self.width, self.height))

    def add_image(self, img:bytes) -> None:
        """Appends a new image to the video
        Each frame is stored in the buffer, so we don't need to save it as a
        file.
        Args:
            img (bytes): Current frame is stored in string buffer
        """
        nparr = np.frombuffer(img, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = cv2.resize(img_np, 
                            (self.width, self.height),
                            cv2.INTER_LANCZOS4)
        self.video.write(frame)

    def close_recording(self) -> None:
        """Stops the recording
        Raises:
            Exception:
                If no recording has been initiated, the video can't be released
        """
        if self.video is None:
            raise Exception('No video initialized.')
        self.video.release()
        cv2.destroyAllWindows()