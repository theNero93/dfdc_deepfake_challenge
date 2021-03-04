import argparse
import os
import re

import numpy as np
import torch
from PIL import Image
from albumentations import image_compression
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.basic_dataset import BasicDataset
from src.model.transforms.transforms import create_basic_transforms
from kernel_utils import isotropically_resize_image, put_to_center, normalize_transform
from src.util.validate import calc_scores
from training.zoo.classifiers import DeepFakeClassifier


def extract_faces(detector, image):
    h, w = image.shape[:2]
    img = Image.fromarray(image.astype(np.uint8))
    img = img.resize(size=[s // 2 for s in img.size])

    batch_boxes, probs = detector.detect(img, landmarks=False)
    faces = []
    scores = []
    for bbox, score in zip(batch_boxes, probs):
        if bbox is not None:
            xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox]
            w = xmax - xmin
            h = ymax - ymin
            p_h = h // 3
            p_w = w // 3
            crop = image[max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
            faces.append(crop)
            scores.append(score)

    frame_dict = {"video_idx": 0,
                  "frame_idx": 0,
                  "frame_w": w,
                  "frame_h": h,
                  "faces": faces,
                  "scores": scores}
    return frame_dict


def pred_on_image(models, detector, image, batch_size, input_size, strategy=np.mean, apply_compression=False):
    try:
        faces = extract_faces(detector, image)
        if len(faces) > 0:
            x = np.zeros((batch_size, input_size, input_size, 3), dtype=np.uint8)
            n = 0
            for face in faces["faces"]:
                resized_face = isotropically_resize_image(face, input_size)
                resized_face = put_to_center(resized_face, input_size)
                if apply_compression:
                    resized_face = image_compression(resized_face, quality=90, image_type=".jpg")
                if n + 1 <= batch_size:
                    x[n] = resized_face
                    n += 1
                else:
                    pass
            if n > 0:
                x = torch.tensor(x, device="cuda").float()
                # Preprocess the images.
                x = x.permute((0, 3, 1, 2))
                for i in range(len(x)):
                    x[i] = normalize_transform(x[i] / 255.)
                # Make a prediction, then take the average.
                with torch.no_grad():
                    preds = []
                    for model in models:
                        y_pred = model(x[:n].half())
                        y_pred = torch.sigmoid(y_pred.squeeze())
                        bpred = y_pred.cpu().numpy()
                        preds.append(strategy(bpred))
                    return np.mean(preds)
    except Exception as e:
        print("Prediction error on video %s: %s" % (0, str(e)))

    return 0.5


def test(args):
    models = []
    model_paths = [os.path.join(args.weights_dir, model) for model in args.models]
    frames_per_video = 1
    input_size = 380
    for path in model_paths:
        model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cuda")
        print("loading state dict {}".format(path))
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=False)
        model.eval()
        del checkpoint
        models.append(model.half())

    normalize = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    test_data = BasicDataset(root_dir=args.root_dir,
                             processed_dir=args.processed_dir,
                             crops_dir=args.crops_dir,
                             split_csv=args.split_csv,
                             seed=args.seed,
                             normalize=normalize,
                             transforms=create_basic_transforms(args.size),
                             mode='test')
    test_loader = DataLoader(test_data, batch_size=1)

    y_pred, y_true = [], []
    for img, label in tqdm(test_loader):
        with torch.no_grad():
            preds = []
            for model in models:
                y_pred = model(img.cuda())
                y_pred = torch.sigmoid(y_pred.squeeze())
                bpred = y_pred.cpu().numpy()
                preds.append(np.mean(bpred))
            y_pred.append(np.mean(preds))
            y_true.append(y_pred)

    acc, ap, auc = calc_scores(y_true, y_pred)[:3]
    print("Test: acc: {}; ap: {}; auc: {}".format(acc, ap, auc))


def parse_args():
    parser = argparse.ArgumentParser(description="Parameters for Training")
    args = parser.add_argument
    # Dataset Options
    args("--root_dir", default='datasets/dfdc', help="root directory")
    args('--processed_dir', default='processed', help='directory where the processed files are stored')
    args('--crops_dir', default='crops', help='directory of the crops')
    args('--split_csv', default='folds.csv', help='Split CSV Filename')
    args('--seed', default=111, help='Random Seed')
    args('--weights-dir', type=str, default="./src/baselines/dfdc_winner/weights",
         help="path to directory with checkpoints")
    args('--models', nargs='+', required=True, help="checkpoint files")
    args('--size', default=380)
    return parser.parse_args()


if __name__ == '__main__':
    test(parse_args())
