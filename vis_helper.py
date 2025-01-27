import os
from abc import ABC

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from torchvision.transforms import transforms


class Visualizer(ABC):
    def __init__(
        self,
        vis_dir="/home/mgx/project/FFMC/vis/withoutimage2",
        img_dir="/home/mgx/project/Bilinear-Matching-Network-main/dataset/images_384x576",
        activation="sigmoid",
        normalization=True,
        with_image=False,
    ):
        """
        vis_dir: dir to save the visualization results
        img_dir: dir of img
        normalization: if True, the heatmap 1). rescale to [0,1], 2). * 255, 3). visualize.
                       if False, the heatmap 1). * 255, 2). visualize.
        with_image: if True, the image & heatmap would be combined to visualize.
                    if False, only the heatmap would be visualized.
        """
        self.vis_dir = vis_dir
        self.img_dir = img_dir
        self.activation_fn = (
            self.build_activation_fn(activation) if activation else None
        )
        self.normalization = normalization
        self.with_image = with_image

    def build_activation_fn(self, activation):
        if activation == "sigmoid":

            def _sigmoid(x):
                return 1 / (1 + np.exp(-x))

            return _sigmoid
        else:
            raise NotImplementedError

    def apply_scoremap(self, image, scoremap, alpha=0.5):
        # scoremap = gaussian_filter(scoremap, 10)
        np_image = np.asarray(image, dtype=float)
        scoremap = (scoremap * 255).astype(np.uint8)
        scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
        scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
        return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

    def vis_result(self, imgid, resname, output):
        """
        filename: str
        image: tensor c x h x w
        """
        os.makedirs(self.vis_dir, exist_ok=True)
        filepath = os.path.join(self.vis_dir, resname)
        filename = imgid+".jpg"
        output = output.permute(1, 2, 0)  # c x h x w -> h x w x c
        output = output.cpu().detach().numpy()
        if self.activation_fn:
            output = self.activation_fn(output)
        if self.normalization:
            output = (output - output.min()) / (output.max() - output.min())
        if self.with_image:
            img_path = os.path.join(self.img_dir, filename)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            output = self.apply_scoremap(image, output)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        else:
            output = (output * 255).astype(np.uint8)
        cv2.imwrite(filepath, output)

    def vis_batch(self, imgids, targets, outputs):


        for (imgid, target, output) in zip(
            imgids, targets, outputs
        ):

            cnt_gt = int(targets.sum().round())
            cnt_pred = round(output.sum().item(), 1)
            resname = "{}_gt{}_pred{}.png".format(imgid, cnt_gt, cnt_pred)
            self.vis_result(imgid, resname, output)



