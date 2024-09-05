import os
import numpy as np
from PIL import Image
import torch
import cv2
import sys
import cv2
import os
import re
import numpy as np
from tqdm import tqdm
from sklearn.metrics import jaccard_score, accuracy_score
import torch.nn.functional as F
dataset_name = sys.argv[1]




def resize_mask(mask, target_shape):
    return np.array(Image.fromarray(mask).resize((target_shape[1], target_shape[0]), resample=Image.NEAREST))

def load_mask(mask_path):
    """Load the mask from the given path."""
    return np.array(Image.open(mask_path).convert('L'))  # Convert to grayscale

def region_segment(model, boxes, image=None, image_shape=None, box_instance_feature=None, sam=True):
    t1 = time.time()
    if sam:
        model.set_image(image)
        boxes = np.array(boxes)
        mask, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes[None, :],
        multimask_output=False,
        )
        mask = mask[0, ...]
    else:
        mask_h = boxes[3] - boxes[1]
        mask_w = boxes[2] - boxes[0]
        pred_box_mask = F.interpolate(model(box_instance_feature).unsqueeze(1), mode='bilinear', size=(mask_h, mask_w))
        pred_box_mask = pred_box_mask[0, 0, ...] > 0.5
        mask = torch.zeros(image_shape, dtype=torch.bool, device='cuda')
        mask[boxes[1]:boxes[3], boxes[0]:boxes[2]] = pred_box_mask

        
    return mask, time.time() - t1

def hex_to_rgb(x):
    return [int(x[i:i + 2], 16) / 255 for i in (1, 3, 5)]

class DistinctColors:

    def __init__(self):
        colors = [
            '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f55031', '#911eb4', '#42d4f4', '#bfef45', '#fabed4', '#469990',
            '#dcb1ff', '#404E55', '#fffac8', '#809900', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#f032e6',
            '#806020', '#ffffff',

            "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0030ED", "#3A2465", "#34362D", "#B4A8BD", "#0086AA",
            "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81", "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700",

            "#04F757", "#C8A1A1", "#1E6E00",
            "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
            "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
            "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
        ]
        self.hex_colors = colors
        # 0 = crimson / red, 1 = green, 2 = yellow, 3 = blue
        # 4 = orange, 5 = purple, 6 = sky blue, 7 = lime green
        self.colors = [hex_to_rgb(c) for c in colors]
        self.color_assignments = {}
        self.color_ctr = 0
        self.fast_color_index = torch.from_numpy(np.array([hex_to_rgb(colors[i % len(colors)]) for i in range(8096)] + [hex_to_rgb('#000000')]))

    def get_color(self, index, override_color_0=False):
        colors = [x for x in self.hex_colors]
        if override_color_0:
            colors[0] = "#3f3f3f"
        colors = [hex_to_rgb(c) for c in colors]
        if index not in self.color_assignments:
            self.color_assignments[index] = colors[self.color_ctr % len(self.colors)]
            self.color_ctr += 1
        return self.color_assignments[index]

    def get_color_fast_torch(self, index):
        return self.fast_color_index[index]

    def get_color_fast_numpy(self, index, override_color_0=False):
        index = np.array(index).astype(np.int32)
        if override_color_0:
            colors = [x for x in self.hex_colors]
            colors[0] = "#3f3f3f"
            fast_color_index = torch.from_numpy(np.array([hex_to_rgb(colors[i % len(colors)]) for i in range(8096)] + [hex_to_rgb('#000000')]))
            return fast_color_index[index % fast_color_index.shape[0]].numpy()
        else:
            return self.fast_color_index[index % self.fast_color_index.shape[0]].numpy()

    def apply_colors(self, arr):
        out_arr = torch.zeros([arr.shape[0], 3])

        for i in range(arr.shape[0]):
            out_arr[i, :] = torch.tensor(self.get_color(arr[i].item()))
        return out_arr

    def apply_colors_fast_torch(self, arr):
        return self.fast_color_index[arr % self.fast_color_index.shape[0]]

    def apply_colors_fast_numpy(self, arr):
        return self.fast_color_index.numpy()[arr % self.fast_color_index.shape[0]]

def get_boundary_mask(arr, dialation_size=1):
    import cv2
    arr_t, arr_r, arr_b, arr_l = arr[1:, :], arr[:, 1:], arr[:-1, :], arr[:, :-1]
    arr_t_1, arr_r_1, arr_b_1, arr_l_1 = arr[2:, :], arr[:, 2:], arr[:-2, :], arr[:, :-2]
    kernel = np.ones((dialation_size, dialation_size), 'uint8')
    if isinstance(arr, torch.Tensor):
        arr_t = torch.cat([arr_t, arr[-1, :].unsqueeze(0)], dim=0)
        arr_r = torch.cat([arr_r, arr[:, -1].unsqueeze(1)], dim=1)
        arr_b = torch.cat([arr[0, :].unsqueeze(0), arr_b], dim=0)
        arr_l = torch.cat([arr[:, 0].unsqueeze(1), arr_l], dim=1)

        arr_t_1 = torch.cat([arr_t_1, arr[-2, :].unsqueeze(0), arr[-1, :].unsqueeze(0)], dim=0)
        arr_r_1 = torch.cat([arr_r_1, arr[:, -2].unsqueeze(1), arr[:, -1].unsqueeze(1)], dim=1)
        arr_b_1 = torch.cat([arr[0, :].unsqueeze(0), arr[1, :].unsqueeze(0), arr_b_1], dim=0)
        arr_l_1 = torch.cat([arr[:, 0].unsqueeze(1), arr[:, 1].unsqueeze(1), arr_l_1], dim=1)

        boundaries = torch.logical_or(torch.logical_or(torch.logical_or(torch.logical_and(arr_t != arr, arr_t_1 != arr), torch.logical_and(arr_r != arr, arr_r_1 != arr)), torch.logical_and(arr_b != arr, arr_b_1 != arr)), torch.logical_and(arr_l != arr, arr_l_1 != arr))

        boundaries = boundaries.cpu().numpy().astype(np.uint8)
        boundaries = cv2.dilate(boundaries, kernel, iterations=1)
        boundaries = torch.from_numpy(boundaries).to(arr.device)
    else:
        arr_t = np.concatenate([arr_t, arr[-1, :][np.newaxis, :]], axis=0)
        arr_r = np.concatenate([arr_r, arr[:, -1][:, np.newaxis]], axis=1)
        arr_b = np.concatenate([arr[0, :][np.newaxis, :], arr_b], axis=0)
        arr_l = np.concatenate([arr[:, 0][:, np.newaxis], arr_l], axis=1)

        arr_t_1 = np.concatenate([arr_t_1, arr[-2, :][np.newaxis, :], arr[-1, :][np.newaxis, :]], axis=0)
        arr_r_1 = np.concatenate([arr_r_1, arr[:, -2][:, np.newaxis], arr[:, -1][:, np.newaxis]], axis=1)
        arr_b_1 = np.concatenate([arr[0, :][np.newaxis, :], arr[1, :][np.newaxis, :], arr_b_1], axis=0)
        arr_l_1 = np.concatenate([arr[:, 0][:, np.newaxis], arr[:, 1][:, np.newaxis], arr_l_1], axis=1)

        boundaries = np.logical_or(np.logical_or(np.logical_or(np.logical_and(arr_t != arr, arr_t_1 != arr), np.logical_and(arr_r != arr, arr_r_1 != arr)), np.logical_and(arr_b != arr, arr_b_1 != arr)), np.logical_and(arr_l != arr, arr_l_1 != arr)).astype(np.uint8)
        boundaries = cv2.dilate(boundaries, kernel, iterations=1)

    return boundaries
def vis_seg(dc, class_index, H, W, rgb=None, alpha = 0.65):
    segmentation_map = dc.apply_colors_fast_torch(class_index)
    if rgb is not None:
        segmentation_map = segmentation_map * alpha + rgb * (1 - alpha)
    boundaries = get_boundary_mask(class_index.view(H, W))
    segmentation_map = segmentation_map.reshape(H, W, 3)
    segmentation_map[boundaries > 0, :] = 0
    segmentation_map = segmentation_map.detach().numpy().astype(np.float32)
    segmentation_map *= 255.
    segmentation_map = segmentation_map.astype(np.uint8)
    return segmentation_map

def read_segmentation_maps(root_dir, downsample=4):
    segmentation_path = os.path.join(root_dir, 'segmentations')
    classes_file_path = os.path.join(root_dir, 'segmentations', 'classes.txt')
    with open(classes_file_path, 'r') as f:
        classes = f.readlines()
    classes = [class_.strip() for class_ in classes]
    classes.sort()
    # print(classes)
    # get a list of all the folders in the directory
    folders = [f for f in sorted(os.listdir(segmentation_path)) if os.path.isdir(os.path.join(segmentation_path, f))]
    seg_maps = []
    images_tensor = []
    idxes = [] # the idx of the test imgs
    for folder in folders:
        idxes.append(int(folder))  # to get the camera id
        seg_for_one_image = []
        image = torch.from_numpy(np.array(Image.open(os.path.join(root_dir, f'images_{downsample}', f'{folder}.jpg')))) / 255
        images_tensor.append(image)
        for class_name in classes:
            # check if the seg map exists
            seg_path = os.path.join(root_dir, f'segmentations/{folder}/{class_name}.png')
            if not os.path.exists(seg_path):
                raise Exception(f'Image {class_name}.png does not exist')
            img = Image.open(seg_path).convert('L')
            # resize the seg map
            if downsample != 1.0:

                img_wh = (int(img.size[0] / downsample), int(img.size[1] / downsample))
                img = img.resize(img_wh, Image.NEAREST) # [W, H]
            img = (np.array(img) / 255.0).astype(np.int8) # [H, W]
            img = img.flatten() # [H*W]
            seg_for_one_image.append(img)

        seg_for_one_image = np.stack(seg_for_one_image, axis=0)
        seg_for_one_image = seg_for_one_image.transpose(1, 0)
        seg_maps.append(seg_for_one_image)

    seg_maps = np.stack(seg_maps, axis=0) # [n_frame, H*W, n_class]
    return seg_maps, images_tensor

def read_pred_segmentation_maps(root_dir, downsample=1):
    classes = os.listdir(root_dir)
    classes = [class_ for class_ in classes if class_ != 'segmentation_maps']
    image_names = os.listdir(os.path.join(root_dir, classes[0], 'test/mask_0'))
    segmentation_path = os.path.join(root_dir, 'segmentations')
    # classes_file_path = os.path.join(root_dir, 'segmentations', 'classes.txt')
    # with open(classes_file_path, 'r') as f:
    #     classes = f.readlines()
    # classes = [class_.strip() for class_ in classes]
    classes.sort()
    classes = classes[::-1]
    # classes = classes[:2]
    # print(classes)
    # print(classes)
    # get a list of all the folders in the directory
    # folders = [f for f in sorted(os.listdir(segmentation_path)) if os.path.isdir(os.path.join(segmentation_path, f))]
    # seg_maps = []
    # seg_vis_maps = []
    index_maps = []
    # idxes = [] # the idx of the test imgs
    for image_name in image_names:
        # idxes.append(int(folder))  # to get the camera id
        seg_for_one_image = []
        index_map = np.full((756, 1008), fill_value=len(classes) - 1, dtype=np.int64)
        # index_map = np.full()
        # print(index_map.shape)
        # index_map = len(classes)
        for i, class_name in enumerate(classes):
            # check if the seg map exists
            # seg_path = os.path.join(root_dir, f'segmentations/{folder}/{class_name}.png')
            seg_path = os.path.join(root_dir, class_name, f'test/mask_0/{image_name}')
            # print(seg_path)
            if not os.path.exists(seg_path):
                raise Exception(f'Image {class_name}.png does not exist')
            img = Image.open(seg_path).convert('L')
            # resize the seg map
            # if downsample != 1.0:

            #     img_wh = (int(img.size[0] / downsample), int(img.size[1] / downsample))
            # img = img.resize(img_wh, Image.NEAREST) # [W, H]
            img = img.resize((1008, 756), Image.NEAREST)
            img = np.array(img)
            # img = (np.array(img) / 255.0).astype(np.int8) # [H, W]
            mask = img > 0

            # if index_map is None:
                
            print(mask.shape)
            # print(index_map.shape)
            index_map[mask] = len(classes) -1 - i
            # img = img.flatten() # [H*W]
            # seg_for_one_image.append(img
        index_maps.append(index_map)

        # seg_for_one_image = np.stack(seg_for_one_image, axis=0)
        # seg_for_one_image = seg_for_one_image.transpose(1, 0)
        # seg_maps.append(seg_for_one_image)

    # seg_maps = np.stack(seg_maps, axis=0) # [n_frame, H*W, n_class]
    return index_maps


iou_scores = {}  # Store IoU scores for each class
biou_scores = {}
class_counts = {}  # Count the number of times each class appears



#prompt_dict_ ={}
# prompt_dict_teatime:"apple":"which is red fruit","bag of cookies":"which is the brown bag on the side of the plate","coffee mug":"which cup is used for coffee","cookies on a plate":"which are the cookies","paper napkin":"what can be used to wipe hands","plate":"what can be used to hold cookies","sheep":"which is a cute white doll","spoon handle":"which is spoon handle","stuffed bear":"which is the brown bear doll","tea in a glass":"which is the drink in the transparent glass"                             
# prompt_dict_figurines:"green apple":"what is green fruit","green toy chair":"what is suitable for people to sit down and is green","old camera":"what can be used to take pictures and is black","porcelain hand":"what is like a part of a person","red apple":"what is red fruit","red toy chair":"what is suitable for people to sit down and is red","rubber duck with red hat":"which is the small yellow rubber duck"
# prompt_dict_ramen:"chopsticks":"which one is the chopstic on the side of yellow bowl","egg":"what is the round, golden, protein-rich object in the bowl","glass of water":"which one is a transparent cup with water in it", "pork belly":"which is the big piece of meat in the bowl", "wavy noodles in bowl":"which are long and thin noodles","yellow bowl":"which is the yellow bowl used to hold noodles"
gt_folder_path = os.path.join('data','ovs3d', dataset_name)
# You can change pred_folder_path to your output
pred_folder_path = os.path.join('ovs3d_masks', f'{dataset_name}')
segmentation_map_path = os.path.join('ovs3d_masks', f'{dataset_name}', 'segmentation_maps')
os.makedirs(segmentation_map_path, exist_ok=True)
gt_seg_maps, images_tensor = read_segmentation_maps(gt_folder_path)
index_maps = read_pred_segmentation_maps(pred_folder_path)
IoUs = []
accuracies = []
dc = DistinctColors()
for i in range(len(gt_seg_maps)):
    gt_mask = gt_seg_maps[i]
    image_tensor = images_tensor[i]
    h, w = image_tensor.shape[:2]
    # pred_mask = pred_seg_maps[i]
    index_map = torch.from_numpy(index_maps[i])
    # print(index_map.sum())

    # print(gt_mask.shape)
    # print(index_map.shape)
    
    segmentation_map = vis_seg(dc, index_map.reshape(-1), h, w, rgb=image_tensor.reshape(-1, 3))
    Image.fromarray(segmentation_map).save(os.path.join(segmentation_map_path, f'{i}.jpg'))
    one_hot = F.one_hot(index_map.long(), num_classes=gt_mask.shape[-1]) # [N1, n_classes]
    one_hot = one_hot.detach().cpu().numpy().astype(np.int8).reshape(h*w, -1)
    # print(one_hot.shape)
    # if pred_mask.shape != gt_mask.shape:
        # pred_mask = resize_mask(pred_mask, gt_mask.shape)
    IoUs.append(jaccard_score(gt_mask, one_hot, average=None))
    # print('iou for classes:', IoUs[-1], 'mean iou:', np.mean(IoUs[-1]))
    accuracies.append(accuracy_score(gt_mask, one_hot))

print(f'iou : {np.mean(IoUs)}  acc: {np.mean(accuracies)}')




# Iterate over each image and category in the GT dataset
# for image_name in os.listdir(gt_folder_path):
#     gt_image_path = os.path.join(gt_folder_path, image_name)
#     # pred_image_path = os.path.join(pred_folder_path, image_name)
    
#     if os.path.isdir(gt_image_path):
#         for cat_file in os.listdir(gt_image_path):
#             cat_id = cat_file.split('.')[0]  # Assuming cat_file format is "cat_id.png"
#             gt_mask_path = os.path.join(gt_image_path, cat_file)
#             pred_mask_path = os.path.join(pred_folder_path, cat_id+'/test/mask_0','test_'+image_name+'.jpg')
            

#             gt_mask = load_mask(gt_mask_path)
#             pred_mask = load_mask(pred_mask_path)
#             print("GT:  ",gt_mask_path)
#             print("Pred:  ",pred_mask_path)

#             if gt_mask is not None and pred_mask is not None:
#                 # Resize prediction mask to match GT mask shape if they are different
#                 if pred_mask.shape != gt_mask.shape:
#                     pred_mask = resize_mask(pred_mask, gt_mask.shape)

#                 iou = calculate_iou(gt_mask, pred_mask)
#                 biou = boundary_iou(gt_mask, pred_mask)
#                 print(iou)
#                 print("IoU: ",iou," BIoU:   ",biou)
#                 if cat_id not in iou_scores:
#                     iou_scores[cat_id] = []
#                     biou_scores[cat_id] = []
#                 iou_scores[cat_id].append(iou)
#                 biou_scores[cat_id].append(biou)
#                 class_counts[cat_id] = class_counts.get(cat_id, 0) + 1

# Calculate mean IoU for each class
# mean_iou_per_class = {cat_id: np.mean(iou_scores[cat_id]) for cat_id in iou_scores}
# mean_biou_per_class = {cat_id: np.mean(biou_scores[cat_id]) for cat_id in biou_scores}

# Calculate overall mean IoU
# overall_mean_iou = np.mean(list(mean_iou_per_class.values()))
# overall_mean_biou = np.mean(list(mean_biou_per_class.values()))

# print("Mean IoU per class:", mean_iou_per_class)
# print("Mean Boundary IoU per class:", mean_biou_per_class)
# print("Overall Mean IoU:", overall_mean_iou)
# print("Overall Boundary Mean IoU:", overall_mean_biou)