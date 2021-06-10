import os
import sys
from math import floor
import cv2
import numpy as np
from .video_utils import (VideoIterator, second_to_ms, ms_to_second, VideoCap)


def compute_extended_start_end(start_msec, end_msec, max_end_msec,
                               fraction_extra):
    """Given the start and end time, compute the new_start_msec
        and new_end_msec such that the clip specified by
        them is fraction_extra more than the original clip
        specified by start_msec and end_msec given that
        the calculated bounds are within the bounds of the video

    As an example:
        if end_msec - start_msec is 100 then
        new_end_msec - new_start_msec is 150
        if fraction_extra is 0.5
    """
    duration = end_msec - start_msec
    msec_extra = duration * fraction_extra / 2

    # prevent our adding from going over
    new_end_msec = min(end_msec + msec_extra, max_end_msec)

    # prevent our subtracting from going under
    new_start_msec = max(0, start_msec - msec_extra)
    return new_start_msec, new_end_msec


class VideoReader:
    """A class for ease of use in reading from the VideoCap class
    """

    def __init__(self, path, load_as_rgb=True):
        """
        Arguments:
            path (str): The path to the video file
            load_as_rgb (bool, optional): To load the image in RGB ordering
                or not
        """
        self.video_cap = VideoCap(path)
        self.load_as_rgb = load_as_rgb
        self.path = path

    @property
    def fps(self):
        """FPS of the video
        """
        return self.video_cap.fps

    @property
    def end_msec(self):
        """End time of the video in msec
        """
        return self.video_cap.end_msec

    @property
    def tot_frames(self):
        """Total number of frames in the video
        """
        return self.video_cap.tot_frames

    @property
    def rounded_fps(self):
        """FPS but rounded
        """
        return self.video_cap.rounded_fps

    def to_iter(self,
                fps=None,
                frame_interval=1,
                start_second=0,
                end_second=None,
                round_source_fps=True,
                max_frames=sys.maxsize,
                fraction_extra=0):
        """
        Arguments:
            fps (float, optional): The FPS to sample the video at,
                must be less or equal to the video's intrinsic FPS
            frame_interval (int): The interval in which we sample frames at,
                so if frame_interval = 10, then only every 10 frames at the
                specified fps will be returned
            start_second (float, optional): The time to start
                sampling the frames at
            end_second (float, optional): The time at which we stop
                sampling the frames, this bound is EXCLUSIVE
            round_source_fps (bool, optional): To round the source FPS to
                an integer or not. Used for when videos have FPS of
                29.97 to round it up to 30
            max_frames (int, optional): maximum number of frames to output
            fraction_extra (float, optional): the fraction of extra time
                we add to the clip specified by start_second and end_second

                As an example:
                    if fraction_extra is 0.5 then the total length
                    of the clip will be 50% greater than the clip specified by
                    start_second and end_second as long as the video has
                    enough frames to support this(i.e we clip the ends
                    and do not pad the video)

        Returns:
            A VideoIterator instance
        """
        source_fps = self.rounded_fps if round_source_fps else self.fps
        target_fps = fps if fps is not None else source_fps

        end_msec = second_to_ms(
            end_second) if end_second is not None else self.end_msec
        start_msec = second_to_ms(start_second)

        if end_msec > self.end_msec:
            raise ValueError(
                f"{self.path}: Specified end time {end_msec} msec is greater" +
                f" than the video's actual end time {self.end_msec} msec")
        if start_msec > end_msec:
            raise ValueError(
                f"{self.path}: Specified start time {start_msec} msec is" +
                " greater than the video's actual end time" +
                f" {self.end_msec} msec")

        start_msec, end_msec = compute_extended_start_end(
            start_msec, end_msec, self.end_msec, fraction_extra)

        # calculate rate to sample relative to the source fps
        interval = source_fps / target_fps * frame_interval

        # calculate max number of frames to sample
        duration_s = ms_to_second(end_msec - start_msec)
        tot_frames_to_sample = min(
            floor(duration_s * target_fps / frame_interval), max_frames)

        iterator = VideoIterator(self.video_cap, start_msec,
                                 tot_frames_to_sample, interval,
                                 self.load_as_rgb)
        return iterator
