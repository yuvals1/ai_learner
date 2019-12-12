import cv2
from os import listdir, makedirs
from os.path import basename, join, exists, dirname
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os


def get_img(img_name, folder_name, dataset, folder_path='/home/mmm/Desktop/XView_Project/data/images'):
    """
    Return numpy image based on image name and folder.
    """
    img_path = join(folder_path, dataset, folder_name, img_name)
    assert os.path.exists(img_path), f'img path not exist: {img_path}'

    if 'mask' in folder_name:
        img = cv2.imread(img_path, 0)
    else:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def torch2numpy(img):
    img = img.permute(1, 2, 0)
    img = img.detach().numpy()
    return img


class ImagesDataset(Dataset):

    def __init__(self, csv_file, data_dir, input_img_folders):
        super().__init__()
        if type(csv_file) == str:
            self.df = pd.read_csv(csv_file)
        else:
            self.df = csv_file
        self.data_dir = data_dir
        self.input_img_types = input_img_folders

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        meta = self.df.iloc[idx]
        sample = dict()
        for img_type in self.input_img_types:
            img_name = meta['img_name']
            dataset = meta['dataset']
            sample[img_type] = get_img(img_name, folder_name=img_type, dataset=dataset, folder_path=self.data_dir)

        x_tup = tuple([sample[k] for k in self.input_img_types])

        return x_tup, meta['img_name'], meta['dataset']


class PredictionsDataset(Dataset):

    def __init__(self, csv_file, data_dir):
        super().__init__()
        if type(csv_file) == str:
            self.df = pd.read_csv(csv_file)
        else:
            self.df = csv_file
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        meta = self.df.iloc[idx]
        sample = None
        img_name = meta['img_name']
        img_dir = join(self.data_dir, img_name)
        for prediction_name in listdir(img_dir):
            prediction = np.load(join(img_dir, prediction_name))
            if sample is None:
                sample = prediction
            else:
                sample = np.append(sample, prediction, axis=2)
        return sample, meta['img_name'], meta['dataset']


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
    ALLOWED_ANGLES = [90, 180, 270]

    def __init__(self, angle):
        super(AugmentRotate, self).__init__(f'Rotate_{angle}')
        self.angle = angle

    def augment(self, tensor, reverse=False):
        if reverse:
            angle = 360 - self.angle
        else:
            angle = self.angle
        assert angle in AugmentRotate.ALLOWED_ANGLES
        if angle == 90:
            tensor = tensor.permute(0, 1, 3, 2).flip(2)
        elif angle == 180:
            tensor = tensor.flip(2).flip(3)
        elif angle == 270:
            tensor = tensor.permute(0, 1, 3, 2).flip(3)
        return tensor

    def reverse_augment(self, tensor):
        return self.augment(tensor, reverse=True)


class AugmentFlip(AugmentWrapper):
    def __init__(self):
        super(AugmentFlip, self).__init__(f'Flip')

    def augment(self, tensor):
        tensor = tensor.flip(2)
        return tensor

    def reverse_augment(self, tensor):
        tensor = tensor.flip(2)
        return tensor


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
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    BASE_AGUMENTS = [ComplexAugment([]),
                     AugmentFlip(),
                     AugmentRotate(90),
                     AugmentRotate(180),
                     AugmentRotate(270),
                     ComplexAugment([AugmentRotate(90), AugmentFlip()]),
                     ComplexAugment([AugmentRotate(180), AugmentFlip()]),
                     ComplexAugment([AugmentRotate(270), AugmentFlip()]),
                     ]

    def __init__(self, models, predictions_dir, df_path, data_dir, input_img_folders,
                 batch_size=1, num_workers=0, ttas=[], cuda_flag=False):
        self.models = models
        self.device = 'cuda' if cuda_flag else 'cpu'
        for model_name in self.models:
            self.models[model_name].eval()
            self.models[model_name].to(self.device)
        self.ttas = ttas
        self.predictions_dir = predictions_dir

        self.dataset = ImagesDataset(csv_file=df_path, data_dir=data_dir,
                                     input_img_folders=input_img_folders)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers)
        self.transformations = [lambda x: x.to(self.device).float(),
                                lambda x: x/255, #Normalization
                                lambda x: x.permute(0, 3, 1, 2), #Change shape from h*w*c to c*h*w
                                lambda x: (x - Inferer.IMAGENET_MEAN) / Inferer.IMAGENET_STD
                                ]

    def transform_tensor(self, tensor):
        for transformation in self.transformations:
            tensor = transformation(tensor)
        return tensor

    def infer_all_models(self):
        softmax = torch.nn.Softmax(dim=1).to(self.device)
        for model_name, model in self.models.items():
            for batch in self.dataloader:
                batch_imgs = [self.transform_tensor(x) for x in batch[0]]
                for augmentation in self.ttas:
                    prediction_paths = [join(self.predictions_dir, img_name, f'{model_name}_{augmentation}')
                                       for img_name in batch[1]]
                    if not all([exists(f'{x}.npy') for x in prediction_paths]):
                        augmented_imgs = [augmentation.augment(img) for img in batch_imgs]
                        with torch.no_grad():
                            predictions = self.models[model_name](augmented_imgs)
                        predictions = softmax(predictions)
                        predictions = augmentation.reverse_augment(predictions)
                        for i in range(len(prediction_paths)):
                            prediction_path = prediction_paths[i]
                            prediction_dir = dirname(prediction_path)
                            if not exists(prediction_dir):
                                makedirs(prediction_dir)
                            np.save(prediction_path, torch2numpy(predictions[i]))


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
            ensemble_pred = self.ensemble(img_preds)
            self.save_prediction(ensemble_pred, img_name)

    def ensemble(self, preds):
        NotImplemented

    def save_prediction(self, ensemble_pred, img_name):
        ensemble_pred = np.argmax(ensemble_pred, axis=2)
        pred_loc = self.post_process(ensemble_pred > 0)
        pred_dmg = ensemble_pred
        cv2.imwrite(join(self.outs_dir, f'{img_name}_localization.png'), pred_loc)
        cv2.imwrite(join(self.outs_dir, f'{img_name}_damage.png'), pred_dmg)

    def post_process(self, probability_img, min_size=0):
        """
        Post processing of each predicted mask, components with lesser number of pixels
        than `min_size` are ignored
        """
        num_component, component = cv2.connectedComponents(probability_img.astype(np.uint8))
        predictions = np.zeros((1024, 1024), np.float32)
        for c in range(1, num_component):
            p = (component == c)
            if p.sum() > min_size:
                predictions[p] = 1
        return predictions


class BasicEnsembler(Ensembler):

    def ensemble(self, preds):
        ensemble_pred = np.mean(preds, axis=0)
        return ensemble_pred


class StackingEnsembler(Ensembler):
    def __init__(self, model, test_dir, test_df_path, outs_dir, batch_size=1, num_workers=0):
        super(StackingEnsembler, self).__init__(test_dir, outs_dir)
        self.model = model
        self.dataset = PredictionsDataset(csv_file=test_df_path, data_dir=test_dir)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, num_workers=num_workers)
        self.transformations = [lambda x: x.to(self.device).float(),
                                lambda x: x.permute(0, 3, 1, 2),  # Change shape from h*w*c to c*h*w
                                ]

    def transform_tensor(self, tensor):
        for transformation in self.transformations:
            tensor = transformation(tensor)
        return tensor

    def predict_all(self):
        for batch in self.dataloader:
            predictions = self.ensemble(batch[0])
            img_names = batch[1]
            for i in range(len(img_names)):
                self.save_prediction(torch2numpy(predictions[i]), img_names[i])

    def ensemble(self, preds):
        preds = self.transform_tensor(preds)
        with torch.no_grad():
            return self.model(preds)


class AAAREnsembler(Ensembler):
    epsilon = 1e-6

    def __init__(self, inps_dir, outs_dir, clustering_model=None, kernel_betta=0):
        super(AAAREnsembler, self).__init__(inps_dir, outs_dir)
        if clustering_model is None:
            self.clustering_model = DBSCAN(eps=3, min_samples=5, metric='precomputed', n_jobs=-1)
        else:
            self.clustering_model = clustering_model
        self.kernel_betta = kernel_betta

    @staticmethod
    def mask_iou(mask1, mask2):
        union = np.sum(np.max(np.array([mask1, mask2]), axis=0))
        intersection = np.sum(mask1 * mask2)
        return intersection/union

    def create_dist_matrix(self, preds):
        preds_num = preds.shape[0]
        dist_mat = np.zeros((preds_num, preds_num))
        for i in range(preds_num):
            for j in range(i, preds_num):
                iou = AAAREnsembler.mask_iou(preds[i], preds[j])
                dist_mat[i, j] = iou
                dist_mat[j, i] = iou
        dist_mat = self.sim2dist(dist_mat)
        return dist_mat

    def sim2dist(self, x):
        if self.kernel_betta:
            return np.exp(-self.kernel_betta*x)
        return 1/(AAAREnsembler.epsilon + x)

    def cluster_predictions(self, preds):
        dist_matrix = self.create_dist_matrix(preds)
        clustering = self.clustering_model.fit(dist_matrix)
        return np.array(clustering.labels_)

    def ensemble(self, preds):
        clusters = self.cluster_predictions(preds)
        clusters_set = set(clusters)
        pred = np.zeros(preds[0].shape)
        for cluster in clusters_set:
            pred += np.mean(pred[clusters == cluster], axis=0)
        pred = pred / len(clusters_set)
        return pred