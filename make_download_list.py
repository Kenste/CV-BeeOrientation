import glob, os, random

# 1) Point to your .txt folder
ann_dir  = "annotations/30fps/frames_txt"
# 2) Base URL for the 30 fps frames
base_url = "https://beepositions.unit.oist.jp/30fps/"

urls = []
for txt in sorted(glob.glob(f"{ann_dir}/frame_*.txt")):
    with open(txt) as f:
        # parse lines: offset_x offset_y class pos_x pos_y angle
        lines = [l.split() for l in f if len(l.split())==6]
    # if any fully visible bee (class == "1")
    if any(int(parts[2]) == 1 for parts in lines):
        fn = os.path.basename(txt).replace(".txt", ".png")
        urls.append(base_url + fn)

# shuffle & take e.g. 3000 frames
random.seed(42)
random.shuffle(urls)
urls = urls[:3000]

# write to file
with open("to_download.txt", "w") as f:
    f.write("\n".join(urls))
print(f"Wrote {len(urls)} URLs to to_download.txt")
