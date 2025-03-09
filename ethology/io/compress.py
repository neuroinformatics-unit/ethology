import typing

import ffmpeg
from movement import sample_data


def compress_video_h264(video_path:str, 
                  saveout_path:str, 
                  crf: int = 23,
) -> None:  
    """ Compresses and optionally trims the local video, and saves out. 

    Args:
        video_path (str): local video filepath to be trimmed
        saveout_path (str): local filepath for saving file to
        crf (int, optional): Compression rate factor. Defaults to 23.
    """
    input = ffmpeg.input(video_path)
    output = input.output(saveout_path, vcodec='libx264', pix_fmt='yuv420p', preset='superfast', crf=crf)
    output.run(overwrite_output = True)


