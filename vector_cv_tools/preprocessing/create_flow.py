from multiprocessing import Pool
import subprocess as sp
import logging
import pathlib
import argparse
import inspect
import os
from functools import partial

import cv2
import tqdm
import numpy as np
import tempfile

from vector_cv_tools.utils import VideoCap, bgr2rgb

# Dense flow algorithms derived from
# https://docs.opencv.org/4.1.0/df/dde/classcv_1_1DenseOpticalFlow.html

CV2_DENSEFLOW_MAP = {
    "DIS": cv2.DISOpticalFlow_create,
    "Farneback": cv2.FarnebackOpticalFlow_create,
    "DenseRLOF": cv2.optflow.DenseRLOFOpticalFlow_create,
    "DualTVL1": cv2.optflow.DualTVL1OpticalFlow_create,
    "PCAFlow": cv2.optflow_OpticalFlowPCAFlow,
    "VariationalRefinement": cv2.VariationalRefinement_create,
}


class FFMEGWriter():

    def __init__(self, outf, fps, width, height):

        self.outf = outf
        self.dimensions = "{}x{}".format(width, height)
        self.fps = fps

        self.command = [
            "ffmpeg",
            '-f',
            'rawvideo',  # video is not in any container
            '-vcodec',
            'rawvideo',  # video is not in any container
            '-s',
            self.dimensions,  # dimensions
            '-pix_fmt',
            'rgb24',  # 3 channels of int8
            '-r',
            str(self.fps),  # framerate
            '-i',
            '-',  # input from stdin
            '-an',  # audio no
            '-vcodec',
            'mpeg4',
            '-b:v',
            '5000k',  # bit rate: to a high enough number
            self.outf
        ]

        self.proc = sp.Popen(self.command,
                             stdin=sp.PIPE,
                             stdout=sp.PIPE,
                             stderr=sp.PIPE)

    def write(self, frame):
        self.proc.stdin.write(frame.tostring())

    def release(self):
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.proc.stderr.close()
        self.proc.wait()

    def stdout(self):
        return self.proc.stdout.read().decode()

    def stderr(self):
        return self.proc.stderr.read().decode()


def norm_quantize_and_fill_rgb(optical_flow):
    # This is actually -128 to 127, but using 127 to make normalization symmetric
    INT8_POS_MAX = 127

    max_pos = max(np.max(optical_flow), 0)
    max_neg = -1 * min(np.min(optical_flow), 0)
    norm_max = max(max_pos, max_neg)

    if norm_max > 0:
        optical_flow = optical_flow / norm_max

    img_flow = (optical_flow * INT8_POS_MAX).astype(np.int8)

    # currently, the third channel is unused
    img_flow = np.dstack((img_flow, np.zeros(img_flow.shape[:-1],
                                             dtype=np.int8)))
    return img_flow


def get_cv2_flow_algs(module):

    return list(CV2_DENSEFLOW_MAP.keys())


def prepare_args():

    supported_flow_algs = get_cv2_flow_algs(cv2.optflow)

    parser = argparse.ArgumentParser(
        description='creates optical flow based on input folder')

    parser.add_argument('-i',
                        '--input_root_dir',
                        type=str,
                        required=True,
                        help="top directory that contains the video files")

    parser.add_argument('-o', '--output_root_dir',
        type=str,
        default=None,
        help='output directory that contains the flow files, it will '\
             'have the same directory structure of the input directory, except '\
             'for each input')

    parser.add_argument(
        '-e',
        '--extensions',
        type=str,
        default=["mp4"],
        nargs='+',
        help="The extensions to create optical flow for, default is mp4")

    parser.add_argument('-a',
                        '--algorithm',
                        choices=supported_flow_algs,
                        type=str,
                        required=True,
                        help='choice of optical flow algorithms')

    parser.add_argument(
        '-n',
        '--num_workers',
        type=int,
        default=1,
        help='number of processes to use for parallel processing of the videos')

    parser.add_argument('-d', '--define',
        type=str, metavar=('name', 'value'),
        action='append', nargs=2, default=[],
        help='Additional inputs passed to the OpenCV optical_flow algorithm. '\
             'More information can be found at: '\
             'https://docs.opencv.org/3.4/df/dde/classcv_1_1DenseOpticalFlow.html')

    parser.add_argument(
        '--width',
        type=int,
        default=None,
        help='Output width of the flow diagram, default is None (unscaled)')

    parser.add_argument(
        '--height',
        type=int,
        default=None,
        help='Output height of the flow diagram, default is None (unscaled)')

    parser.add_argument(
        '--resize',
        type=float,
        default=None,
        help=
        'Scale factor to resize the video, must be elusive with width and height'
    )

    parser.add_argument(
        "--to_gray",
        default=False,
        action="store_true",
        help=
        "Whether to convert images to grayscale before calculating optical flow, default is False"
    )

    parser.add_argument(
        "--exists_ok",
        default=False,
        action="store_true",
        help="Whether to continue when some path already exists. "\
             "If set to True, the worker will assume the flow video is already "\
             "generated, and returns True, otherwise it will raise ValueError. "\
             "Default is False." )

    args = parser.parse_args()
    if args.output_root_dir is None:
        args.output_root_dir = args.input_root_dir

    if args.resize is not None and \
        (args.width is not None or args.height is not None):
        raise ValueError(
            "Cannot have both --resize and one of -w/--width or -h/--height")

    if (args.width is None) != (args.height is None):
        raise ValueError("--width and --height must be specified together")

    if args.resize is not None and (args.resize <= 0 or args.resize > 1):
        raise ValueError("Invalid resize number {}, must be (0, 1]".format(
            args.resize))

    if args.algorithm == "DenseRLOF" and args.to_gray:
        raise ValueError("Cannot have DenseRLOF with grayscale images!")

    return args


def flow_worker(create_flow_fn, flow_args, args, filename):

    inf, outf = filename
    width, height, resize = args.width, args.height, args.resize

    if os.path.exists(outf):
        if args.exists_ok:
            return outf, True
        else:
            raise ValueError("{} already exists!".format(outf))

    to_gray = args.to_gray

    do_scale = (width is not None and height is not None) or (resize
                                                              is not None)

    flow = create_flow_fn(**flow_args)

    reader = VideoCap(inf)
    ret, previ = reader.read()
    if not ret:
        return outf, False

    if do_scale:
        if resize is not None:
            img_shape = tuple(
                int(l * resize) for l in reversed(previ.shape[:-1]))
        else:
            img_shape = (width, height)
        previ = cv2.resize(previ, img_shape)
    else:
        img_shape = tuple(reversed(previ.shape[:-1]))
    optical_flow = np.zeros(previ.shape[:-1] + (2,), dtype=np.float32)

    if to_gray:
        previ = cv2.cvtColor(previ, cv2.COLOR_BGR2GRAY)

    writer = FFMEGWriter(outf, reader.fps, img_shape[0], img_shape[1])

    success = False
    while True:
        ret, nexti = reader.read()
        if not ret:
            break
        if do_scale:
            nexti = cv2.resize(nexti, img_shape)
        if to_gray:
            nexti = cv2.cvtColor(nexti, cv2.COLOR_BGR2GRAY)

        optical_flow = flow.calc(previ, nexti, optical_flow)
        flow_img = norm_quantize_and_fill_rgb(optical_flow)
        writer.write(flow_img)

        success = True
        previ = nexti

    writer.release()

    return outf, success


def glob_files(base_root, extensions, dest_root):

    logging.info("Looking for files with extensions {}".format(extensions))
    base_len = len(str(base_root)) + 1

    files = []
    for ext in extensions:
        inputs = [str(i) for i in base_root.rglob("*.{}".format(ext))]
        outputs = [
            str((dest_root / i[base_len:]).with_suffix(".flow.avi"))
            for i in inputs
        ]
        for path in outputs:
            parent = os.path.dirname(path)
            os.makedirs(parent, exist_ok=True)
        files.extend(zip(inputs, outputs))

    return files


def processs(args):

    input_dir = pathlib.Path(args.input_root_dir).resolve()
    output_dir = pathlib.Path(args.output_root_dir).resolve()

    logging.info("Processing files with extensions \"{}\" under {}".format(
        args.extensions, input_dir))
    logging.info("Results will be written under {}".format(output_dir))

    files = glob_files(input_dir, args.extensions, output_dir)

    logging.info("Found {} files, processing with {} workers".format(
        len(files), args.num_workers))

    create_flow_fn = CV2_DENSEFLOW_MAP[args.algorithm]
    flow_args = {k: eval(v) for k, v in args.define}
    flow_fn = partial(flow_worker, create_flow_fn, flow_args, args)

    with Pool(args.num_workers) as p:
        for f, res in tqdm.tqdm(p.imap(flow_fn, files), total=len(files)):
            if not res:
                logging.warning(
                    "Unable to generate flow diagram for {}".format(f))

    logging.info("Done!")


def main():

    args = prepare_args()
    FORMAT = '%(asctime)s, [%(filename)s:%(lineno)d] %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logging.info("args: {}".format(args))

    processs(args)


if __name__ == "__main__":
    main()
