import glob
import os
import cv2
import pandas as pd
from tqdm import tqdm

from model import PersonSegmentation

if __name__ == '__main__':
    csv = "cathrynli.csv"
    df = pd.read_csv(csv)
    df.set_index('filename', inplace=True)

    results = {}
    # change 'cpu' to 'cuda' if you have pytorch cuda and your discrete GPU has enough VRAM
    # output size will autoscale to fit input image aspect ratio
    # if you want full image resolution set 'is_resize=False'
    ps = PersonSegmentation('cuda', is_resize=True, resize_size=480)
    output_folder = "out2"
    os.makedirs(output_folder, exist_ok=True)
    images_dir = r"cathrynli"
    images = glob.glob(os.path.join(images_dir, "*.jpg"))

    for img in tqdm(images):
        seg_map = ps.person_segment(img)
        frame = ps.decode_segmap(seg_map, img)
        # skin_frame, skin2img_ratio = ps.skin_segment(frame)
        skin_frame, skin2img_ratio = ps.skin_segment_pro(frame)

        output_file = os.path.join(output_folder, os.path.basename(img)[:-4] + "_out.jpg")
        cv2.imwrite(output_file, cv2.hconcat([frame, skin_frame]))
        results[os.path.basename(img)] = skin2img_ratio

    # add skin to image ratio into a new column according to filename
    df['skin2img ratio'] = df.index.to_series().map(results)
    # overwrite existing csv
    df.to_csv(csv)
