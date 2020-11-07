import cv2
import numpy as np
import torch
from torchvision import models
import torchvision.transforms as T
from PIL import Image


class PersonSegmentation(object):
    def __init__(self, device, is_resize=True, resize_size=480):
        self.device = device
        self.resize_size = resize_size
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()
        self.transform_list = [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])]
        if is_resize:
            self.transform_list.insert(0, T.Resize(self.resize_size))
        self.transform = T.Compose(self.transform_list)

    # use pretrained torchvision deeplabv3 model
    def person_segment(self, filename):
        img = Image.open(filename)

        self.x = self.transform(img).unsqueeze(0).to(self.device)
        self.out_h, self.out_w = self.x.shape[2:]
        # print(f"Output image shape HxW: {self.out_h, self.out_w}")

        out = self.model.to(self.device)(self.x)['out']

        output_map = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        # print(f"Output shape: {out.shape}. 21 channels = 21 classes.")

        return output_map

    # use opencv color range filter
    def skin_segment(self, frame):
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # skin colour range
        min_HSV = np.array([0, 48, 80], dtype="uint8")
        max_HSV = np.array([20, 255, 255], dtype="uint8")

        mask = cv2.inRange(frame_HSV, min_HSV, max_HSV)
        # blur to remove noise
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        frame = cv2.bitwise_and(frame, frame, mask=mask)

        # find the ratio
        skin2img_ratio = np.count_nonzero(mask) / (frame.size / 3)
        return frame, skin2img_ratio

    def skin_segment_pro(self, frame):

        # copy paste
        lower_HSV_values = np.array([0, 40, 0], dtype="uint8")
        upper_HSV_values = np.array([25, 255, 255], dtype="uint8")

        lower_YCbCr_values = np.array((0, 138, 67), dtype="uint8")
        upper_YCbCr_values = np.array((255, 173, 133), dtype="uint8")

        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_YCbCr = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

        # A binary mask is returned. White pixels (255) represent pixels that fall into the upper/lower.
        mask_YCbCr = cv2.inRange(frame_YCbCr, lower_YCbCr_values, upper_YCbCr_values)
        mask_HSV = cv2.inRange(frame_HSV, lower_HSV_values, upper_HSV_values)
        binary_mask_image = cv2.add(mask_HSV, mask_YCbCr)
        image_foreground = cv2.erode(binary_mask_image, None, iterations=3)  # remove noise
        dilated_binary_image = cv2.dilate(binary_mask_image, None,
                                          iterations=3)  # The background region is reduced a little because of the dilate operation
        ret, image_background = cv2.threshold(dilated_binary_image, 1, 128,
                                              cv2.THRESH_BINARY)  # set all background regions to 128

        image_marker = cv2.add(image_foreground,
                               image_background)  # add both foreground and backgroud, forming markers. The markers are "seeds" of the future image regions.
        image_marker32 = np.int32(image_marker)  # convert to 32SC1 format

        cv2.watershed(frame, image_marker32)
        m = cv2.convertScaleAbs(image_marker32)  # convert back to uint8

        m = cv2.GaussianBlur(m, (7, 7), 0)
        # bitwise of the mask with the input image
        _, mask = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        frame = cv2.bitwise_and(frame, frame, mask=mask)

        # find the ratio
        skin2img_ratio = np.count_nonzero(mask) / (frame.size / 3)
        return frame, skin2img_ratio

    def inverse_normalize(self, x, mean, std):
        for t, m, s in zip(x, mean, std):
            t.mul_(s).add_(m)
        return x

    def decode_segmap(self, mask, filename):
        frame = cv2.imread(filename)
        frame = cv2.resize(frame, (self.out_w, self.out_h))
        # 15 is the integer that represents "person" class, there are 21 classes in PascalVOC+background class
        roi = mask == 15
        roi = np.expand_dims(roi, axis=2)
        frame = frame * roi
        return frame
