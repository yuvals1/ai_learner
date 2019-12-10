import cv2
from os import listdir, makedirs
from os.path import basename, join, exists
import numpy as np
import albumentations as albu

class Ensembler(object):
    def _init_(self, models, pre_dir, pred_dir, ttas=[]):
        self.models = models
        self.ttas = ttas
        self.pre_dir = pre_dir
        self.pred_dir = pred_dir

    def predict_all(self):
        for pre_file in listdir(self.pre_dir):
            img, img_name = self.read_data(pre_file)
            preds = self.create_predictions(img, img_name)
            self.basic_ensemble(preds, img_name)

    def read_data(self, pre_file):
        pre, post = cv2.imread(pre_file), cv2.imread(pre_file.replace('pre', 'post'))
        img_name = basename(pre_file.split('.')[0])
        return [pre, post], img_name

    def augment_image(self, img, augment):
        NotImplemented

    def unaugment_pred(self, pred, augment):
        NotImplemented

    def create_predictions(self, img, img_name):
        preds = np.array([])
        for model in self.models:
            for augment in self.ttas:
                img = self.augment_image(img, augment)
                pred_dir = join(self.pred_dir, model, img_name)
                if not exists(pred_dir):
                    makedirs(pred_dir)
                pred_loc = join(pred_dir, augment)
                if exists(pred_loc):
                    pred = np.load(pred_loc)
                else:
                    pred = model(img)
                    pred = self.unaugment_pred(pred, augment)
                    pred = np.save(pred_loc, pred)
                preds.append(pred)
        return preds

    def basic_ensemble(self, preds, img_name):
        pred = np.argmax(np.mean(preds), dim=2)
        pred_loc = self.post_process(pred > 0)
        pred_dmg = pred
        cv2.imwrite(join(self.pred_dir, f'{img_name}_localization.png'), pred_loc)
        cv2.imwrite(join(self.pred_dir, f'{img_name}_damage.png'), pred_dmg)

    def post_process(self, probability_img, min_size=10):
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