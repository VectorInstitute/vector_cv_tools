import os
import warnings
from math import floor
from PIL import Image
from .misc import bgr2rgb
import cv2


def second_to_ms(s):
    return s * 1000


def ms_to_second(ms):
    return ms / 1000


class VideoCap:
    """Wrapper around OpenCV's VideoCapture for ease of use
    """
    FallBackFPS = 30.

    def __init__(self, path):
        """
        Arguments:
            path (str): The path to the video file

        NOTE:
            Frame numbers in this class are 0 indexed
        """
        if not os.path.exists(path):
            raise ValueError(f"The path {path} does not exist")
        self.cap = cv2.VideoCapture(path)
        self.vid_path = path

    @property
    def path(self):
        """Path of the video
        """
        return self.vid_path

    @property
    def height(self):
        """Height of the frames in the video
        """
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def width(self):
        """Width of the frames in the video
        """
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def pos_msec(self):
        """Current position of the video in msec
        """
        return self.cap.get(cv2.CAP_PROP_POS_MSEC)

    @property
    def fps(self):
        """FPS of the video
        """
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        # we need a fallback FPS to deal with many division by zero
        # errors. It would be safe to assume a common FPS in
        # cameras since most videos are taken that way
        if fps <= 0:
            return VideoCap.FallBackFPS
        return fps

    @property
    def rounded_fps(self):
        """Return the rounded FPS of the video
        """
        return round(self.fps)

    @property
    def tot_frames(self):
        """The total number of frames in the video
        """
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def curr_frame(self):
        """Index of the frame to be decoded next
        """
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    @property
    def end_msec(self):
        """The endtime of the video in msec
        """
        return second_to_ms(1 / self.fps * self.tot_frames)

    def read(self):
        """Read a frame using the current VideoCapture state

        Returns:
            tuple: Tuple(load_success, frame) where load_success is True,
                frame is not None if frame is read. Otherwise
                load_success is False and frame is None
        """
        return self.cap.read()

    def set_pos_msec(self, msec):
        """Sets the current position of the video file timestamp to msec

        Arguments:
            msec (float): The time in msec to set the video to

        Returns:
            bool: If set properly this function will return True else False
        """
        return self.cap.set(cv2.CAP_PROP_POS_MSEC, msec)

    def set_curr_frame(self, frame_no):
        """Sets the next frame to decode

        Arguments:
            frame_no (int): The frame number to decode next

        Returns:
            bool: If set properly this function will return True else False
        """
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    def read_frame(self, frame_number):
        """Reads the specified frame_number

        Arguments:
            frame_no (int): The frame to read

        Returns:
            tuple: Tuple(load_success, frame) where load_success indicates if the
                frame is read properly and frame is the image for the specified
                frame number. If the frame is not read correctly then load_success
                will be False and frame will be None
        """
        self.set_curr_frame(frame_number)
        return self.read()

    def check_end_time(self, end_msec):
        """Checks if the current position is BEFORE end_msec

        Arguments:
            end_msec (float): The time in msec we want to check

        Returns:
            bool: True if current video file time position is less than
            end_msec, False otherwise.

        NOTE:
            This bound is exclusive
        """
        return self.pos_msec < end_msec

    def __del__(self):
        self.cap.release()


class VideoIterator:
    """An iterable class that reads frames from a VideoCap, where on error
        there will be a warning thrown and stops the iteration instead of
        the program crashing.

        The length of this object will give the expected frames to be read
            so something have had to gone wrong if the __iter__ call
            nets different number of frames than len(VideoIterator)
    """

    def __init__(self,
                 cap,
                 start_msec,
                 tot_frames_to_sample,
                 interval,
                 load_as_rgb=True):
        """
        Arguments:
            cap (VideoCap): An instance of VideoCap
            start_msec (float): The time to start
                sampling the frames at in msec
            tot_frames_to_sample (int): The total number of frames
                to sample
            interval (float): The computed interval to sample frames at
                relative to the video's innate FPS
            load_as_rgb (bool, optional): To load the image in RGB ordering
                or not
        """
        self.cap = cap
        self.start_msec = start_msec
        self.tot_frames_to_sample = tot_frames_to_sample
        self.interval = interval
        self.load_as_rgb = load_as_rgb

    @property
    def path(self):
        """Path of the video
        """
        return self.cap.path

    @property
    def fps(self):
        """Path of the video
        """
        return self.cap.fps

    def __iter__(self):
        """Stops the iteration on any potential error and throws a warning
        """
        source_frame_count, target_frame_count, load_success = 0, 0, True

        self.cap.set_pos_msec(self.start_msec)
        # sample the frames based on source and target fps
        while load_success and target_frame_count < self.tot_frames_to_sample:
            try:
                load_success, frame = self.cap.read()
            except:
                warnings.warn("Video Loading Failed, stopping iteration \
                              prematurely")

            if frame is not None and load_success:
                while floor(target_frame_count *
                            self.interval) <= source_frame_count:
                    if self.load_as_rgb:
                        frame = bgr2rgb(frame)
                    yield frame
                    target_frame_count += 1

                source_frame_count += 1

    def __len__(self):
        """Returns the total amount of frames that is ***expected*** to be
            returned
        """
        return self.tot_frames_to_sample


def create_GIF(path_to_save, images, msec_per_frame=100):
    """Creates a GIF based on the passed in images list

    Arguments:
        path_to_save (str): The path to save the GIF
        images (list, NumPy Array): If its a numpy array it must be of the
            shape TxHxWxC, if it's a list it must be of length T where
            each element is a NumPy Array of HxWxC
        ms_per_frame (float, optional): How many msec to display each frame

    NOTE:
        Must pass in uint8 image
    """
    if len(images) > 0:
        images = [Image.fromarray(img) for img in images]
        images[0].save(fp=path_to_save,
                       format="GIF",
                       append_images=images[1:],
                       save_all=True,
                       duration=msec_per_frame,
                       loop=0)
    else:
        warnings.warn("Image not saved because an empty list is passed in")
