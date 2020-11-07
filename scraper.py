import csv
import os
from itertools import islice
from math import ceil

from instaloader import Instaloader, Profile


def scrape(name, percentage):
    PROFILE = name  # profile to download from
    X_percentage = percentage  # percentage of posts that should be downloaded

    L = Instaloader()

    profile = Profile.from_username(L.context, PROFILE)
    posts = profile.get_posts()

    posts_sorted_by_likes = sorted(posts,
                                   key=lambda p: p.likes + p.comments,
                                   reverse=True)  # false means that the order is starting from the least popular post

    # need to write number of likes and comments to csv
    os.makedirs(PROFILE, exist_ok=True)

    output_csv = PROFILE + '.csv'
    with open(output_csv, 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["filename", "numOfLikes", "numOfComments", "caption"])

        for idx, post in enumerate(islice(posts_sorted_by_likes, ceil(profile.mediacount * X_percentage / 100))):
            output_filename = os.path.join(PROFILE, PROFILE)
            L.download_pic(output_filename, post.url, post.date_local, str(idx))  # download the picture
            writer.writerow([PROFILE + "_" + str(idx) + ".jpg", post.likes, post.comments, post.caption])


if __name__ == '__main__':
    profile = "cathrynli"
    scrape(profile, 50)  # top 50%
