import argparse
import os
import re
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch
from PIL import Image
from albumentations import image_compression
from facenet_pytorch.models.mtcnn import MTCNN
from tqdm import tqdm

from kernel_utils import isotropically_resize_image, put_to_center, normalize_transform
from training.zoo.classifiers import DeepFakeClassifier
from src.util.dotdict import Dotdict
from src.util.data.data_loader import load_subset
from src.util.validate import calc_scores


def get_opt():
    opt = Dotdict()

    opt.model = 'all'
    opt.is_train = True
    opt.pretrained = True
    opt.checkpoints_dir = './out/checkpoints/faces'
    opt.continue_train = True
    opt.save_name = 'latest'
    opt.name = 'knn'
    # opt.dataset_path = './datasets/celeb-df-v2/images'
    opt.dataset_path = './datasets/forensic/images'
    opt.multiclass = False
    opt.resize_interpolation = 'bilinear'
    opt.load_size = -1
    opt.train_split = 'train'
    opt.train_size = 2500
    opt.val_split = 'val'
    opt.val_size = 100
    opt.test_split = 'test'
    opt.load_value = -1

    return opt


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
    detector = MTCNN(margin=0, thresholds=[0.7, 0.8, 0.8], device="cuda")

    opt = get_opt()
    test_img, test_label = load_subset(opt, opt.test_split, opt.load_value)
    y_pred, y_true = [], []
    for img, label in tqdm(zip(test_img, test_label), total=len(test_img)):
        pred = pred_on_image(models, detector, img, frames_per_video, input_size)
        y_pred.append(pred)
        y_true.append(label)
    acc, ap, auc = calc_scores(y_true, y_pred)[:3]
    print("Test: acc: {}; ap: {}; auc: {}".format(acc, ap, auc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Predict test fotos")
    arg = parser.add_argument
    arg('--weights-dir', type=str, default="./src/baselines/dfdc_winner/weights",
        help="path to directory with checkpoints")
    arg('--models', nargs='+', required=True, help="checkpoint files")
    args = parser.parse_args()
    test(args)
