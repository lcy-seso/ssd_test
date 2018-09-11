import os
import cv2


def show():
    data_dir = "data/0908/resized_images"
    infer_file = "infer_output.txt"

    out_dir = "visual_outputs"
    if not os.path.exists(out_dir): os.mkdir(out_dir)

    path_to_im = dict()

    for line in open(infer_file):
        img_path, _, _, _ = line.strip().split("\t")
        if img_path not in path_to_im:
            im = cv2.imread(os.path.join(data_dir, img_path))
            path_to_im[img_path] = im

    for line in open(infer_file):
        img_path, label, conf, bbox = line.strip().split("\t")
        xmin, ymin, xmax, ymax = map(float, bbox.split(" "))
        xmin = int(round(xmin))
        ymin = int(round(ymin))
        xmax = int(round(xmax))
        ymax = int(round(ymax))

        img = path_to_im[img_path]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, (1 - xmin) * 255, xmin * 255), 2)

    for img_path in path_to_im:
        im = path_to_im[img_path]
        out_path = os.path.join(out_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, im)

    print("Done.")


if __name__ == "__main__":
    show()
