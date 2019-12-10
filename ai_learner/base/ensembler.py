import cv2
from os import listdir, makedirs
from os.path import basename, join, exists
import numpy as np


class AugmentWrapper(object):
    def __init__(self, name):
        self.name = name

    def augment(self, img):
        NotImplemented

    def reverse_augment(self, img):
        NotImplemented

    def __repr__(self):
        return self.name


class AugmentRotate(AugmentWrapper):
    ALLOWED_ANGLES = [0, 90, 180, 270]

    def __init__(self, angle):
        super(AugmentRotate, self).__init__(f'Rotate_{angle}')
        self.angle = angle

    def augment(self, img):
        assert self.angle in AugmentRotate.ALLOWED_ANGLES
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)
        scale = 1.0
        tmp_img = cv2.getRotationMatrix2D(center, self.angle, scale)
        aug_img = cv2.warpAffine(img, tmp_img, (h, w))
        return aug_img

    def reverse_augment(self, img):
        assert self.angle in AugmentRotate.ALLOWED_ANGLES
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)
        scale = 1.0
        tmp_img = cv2.getRotationMatrix2D(center, -self.angle, scale)
        aug_img = cv2.warpAffine(img, tmp_img, (h, w))
        return aug_img


class AugmentFlip(AugmentWrapper):
    def __init__(self):
        super(AugmentFlip, self).__init__(f'Flip')

    def augment(self, img):
        aug_img = cv2.flip(img, 0)
        return aug_img

    def reverse_augment(self, img):
        aug_img = cv2.flip(img, 0)
        return aug_img


class ComplexAugment(AugmentWrapper):
    def __init__(self, augments):
        super(ComplexAugment, self).__init__('_'.join([str(x) for x in augments]))
        self.augments = augments

    def augment(self, img):
        for augment in self.augments:
            img = augment.augment(img)
        return img

    def reverse_augment(self, img):
        for augment in reversed(self.augments):
            img = augment.reverse_augment(img)
        return img


class Inferer(object):
    BASE_AGUMENTS = [ComplexAugment([]),
                     AugmentFlip(),
                     AugmentRotate(90),
                     AugmentRotate(180),
                     AugmentRotate(270),
                     ComplexAugment([AugmentRotate(90), AugmentFlip()]),
                     ComplexAugment([AugmentRotate(180), AugmentFlip()]),
                     ComplexAugment([AugmentRotate(270), AugmentFlip()]),
                     ]

    def __init__(self, models, pre_dir, pred_dir, ttas=[]):
        self.models = models
        self.ttas = ttas
        self.pre_dir = pre_dir
        self.pred_dir = pred_dir

    def infer_all_images(self):
        for pre_file in listdir(self.pre_dir):
            imgs, img_name = self.read_data(join(self.pre_dir, pre_file))
            self.infer_from_models(imgs, img_name)

    def read_data(self, pre_file):
        pre, post = cv2.imread(pre_file), cv2.imread(pre_file.replace('pre', 'post'))
        img_name = basename(pre_file.split('.')[0])
        return [pre, post], img_name

    def infer_from_models(self, imgs, img_name):
        for model in self.models:
            for augment in self.ttas:
                pred_dir = join(self.pred_dir, img_name)
                if not exists(pred_dir):
                    makedirs(pred_dir)
                pred_loc = join(pred_dir, f'{model}_{augment}')
                if not exists(pred_loc):
                    imgs = [augment.augment(x) for x in imgs]
                    pred = model(imgs)
                    pred = augment.reverse_augment(pred)
                    np.save(pred_loc, pred)


class Ensembler(object):
    def __init__(self, inps_dir, outs_dir):
        self.inps_dir = inps_dir
        self.outs_dir = outs_dir

    def predict_all(self):
        for img_name in listdir(self.inps_dir):
            img_dir = join(self.inps_dir, img_name)
            img_preds = []
            for pred_name in listdir(img_dir):
                pred = np.load(join(img_dir, pred_name))
                img_preds.append(pred)
            img_preds = np.array(img_preds)
            self.basic_ensemble(img_preds, img_name)

    def basic_ensemble(self, preds, img_name):
        pred = np.mean(preds, axis=0)
        pred = np.argmax(pred, axis=2)
        pred_loc = self.post_process(pred > 0)
        pred_dmg = pred
        cv2.imwrite(join(self.outs_dir, f'{img_name}_localization.png'), pred_loc*50)
        cv2.imwrite(join(self.outs_dir, f'{img_name}_damage.png'), pred_dmg*50)

    def post_process(self, probability_img, min_size=0):
        """
        Post processing of each predicted mask, components with lesser number of pixels
        than `min_size` are ignored
        """
        num_component, component = cv2.connectedComponents(probability_img.astype(np.uint8))
        predictions = np.zeros((1024, 1024), np.float32)
        num = 0
        for c in range(1, num_component):
            p = (component == c)
            if p.sum() > min_size:
                predictions[p] = 1
                num += 1
        return predictions