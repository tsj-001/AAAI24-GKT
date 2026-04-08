import math

import numpy as np
import torch

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def mixup_data(x, y, alpha=1):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index


def cutmix_data(x, y, beta=1):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if beta > 0:
        lam = np.random.beta(beta, beta)
    else:
        lam = 1

    batch_size = x.size()[0]
    
    index = torch.randperm(batch_size).cuda()

    y_a, y_b = y, y[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x_sliced = x[index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return [bbx1, bby1, bbx2, bby2 ], y_a, y_b, lam, x_sliced, index


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_k_criterion(criterion, pred, ys, lams):
    return sum([lam * criterion(pred, y) for y, lam in zip(ys, lams)])


def mixup_k_data(x, main_biggest=True):
    """ Mix up k pairs, x[0] is the main one, using main coef to distinguish the main one"""
    '''Returns mixed inputs, pairs of targets, lambda and index'''
    k = len(x)
    device = x[0].device
    x = torch.stack(x, dim=0) # (k, bs, h, w)
    lam_list = []
    if main_biggest:
        rand_num = torch.rand(1).item()
        lam = rand_num if rand_num >= 0.5 else (1 - rand_num)
        lam_list.append(lam)
        lam_list += [(1 - lam) / (k - 1)] * (k - 1)
        lam_list = torch.tensor(lam_list, device=device)
    else:
        rand_list = torch.cat([torch.tensor([0], device=device), torch.rand(k-1, device=device), torch.tensor([1], device=device)]).sort()[0]
        lam_list = rand_list[1:] - rand_list[:-1]
        lam_list = lam_list.sort(descending=True)[0]
    mixed_x = (x * lam_list.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)).sum(dim=0)
    return mixed_x, lam_list



""" For cutmix k area """
def cutmix_k_data(x, main_biggest=True, max_iter=1000):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    k = len(x)
    device = x[0].device
    area_x = x[0].size(2) * x[0].size(3)
    x = torch.stack(x, dim=0)  # (k, bs, h, w)

    rectangle = (0, 0, x[0].size(2), x[0].size(3))
    for i in range(max_iter):
        lam_list = []
        if main_biggest:
            rand_num = torch.rand(1).item()
            lam = rand_num if rand_num >= 0.5 else (1 - rand_num)
            lam_list.append(lam)
            lam_list += [(1 - lam) / (k - 1)] * (k - 1)
            lam_list = torch.tensor(lam_list, device=device)
        else:
            rand_list = torch.cat([torch.tensor([0], device=device), torch.rand(k - 1, device=device),
                                   torch.tensor([1], device=device)]).sort()[0]
            lam_list = rand_list[1:] - rand_list[:-1]
            lam_list = lam_list.sort(descending=True)[0]
        area_list = (lam_list * area_x).int()
        area_list[area_list <= 10] = 10
        area_list = area_list.tolist()
        for _ in range(100):
            cut_rectangle_list = cut_areas_from_rectangle(area_list, rectangle)
            if cut_rectangle_list is not None:
                break
        if cut_rectangle_list is not None:
            break
    cut_rectangle_list = torch.tensor(cut_rectangle_list, device=device)
    # 1 is not mask, 0 is mask
    for j in range(k-1):
        x[0][:, :, cut_rectangle_list[j][0]:cut_rectangle_list[j][2], cut_rectangle_list[j][1]:cut_rectangle_list[j][3]] \
            = x[j+1][:, :, cut_rectangle_list[j][0]:cut_rectangle_list[j][2], cut_rectangle_list[j][1]:cut_rectangle_list[j][3]]
    return x[0], lam_list



def rectangle_area(rectangle):
    return (rectangle[2] - rectangle[0]) * (rectangle[3] - rectangle[1])


def cut_three_same_rectangle(origin, cut):
    origin = np.array(origin)
    cut = np.array(cut)
    residual_hw = origin - cut
    if residual_hw[0] != 0:
        return [(origin[0], origin[1], cut[0], origin[3])]
    elif residual_hw[1] != 0:
        return [(origin[0], origin[1], origin[2], cut[1])]
    elif residual_hw[2] != 0:
        return [(cut[2], cut[1], origin[2], origin[3])]
    elif residual_hw[3] != 0:
        return [(cut[0], cut[3], origin[2], origin[3])]
    else:
        return []


def get_rest_rectangle_list(origin, cut):
    origin = np.array(origin)
    cut = np.array(cut)
    residual_hw = origin - cut
    is_zeros = residual_hw == 0
    zeros_num = is_zeros.sum()
    if zeros_num == 4:
        return []
    elif zeros_num == 3:
        return cut_three_same_rectangle(origin, cut)
    elif (is_zeros == [False, True, False, True]).all():
        return [(origin[0], origin[1], cut[0], origin[3]), (cut[2], origin[1], origin[2], origin[3])]
    elif (is_zeros == [True, False, True, False]).all():
        return [(origin[0], origin[1], origin[2], cut[1]), (origin[0], cut[3], origin[2], origin[3])]
    else:
        residual_hw_abs = np.abs(residual_hw)
        H = origin[3] - origin[1]
        W = origin[2] - origin[0]
        candidate_area = residual_hw_abs * np.array([H, W, H, W])
        max_area_index = candidate_area.argmax()
        if max_area_index == 0:
            cut_one = (origin[0], origin[1], cut[0], origin[3])
        elif max_area_index == 1:
            cut_one = (origin[0], origin[1], origin[2], cut[1])
        elif max_area_index == 2:
            cut_one = (cut[2], origin[1], origin[2], origin[3])
        else:
            cut_one = (origin[0], cut[3], origin[2], origin[3])
        return [cut_one] + get_rest_rectangle_list(cut_three_same_rectangle(origin, cut_one)[0], cut)


def random_select_value(min_value, max_value):
    if min_value >= max_value:
        return max_value
    return np.random.randint(min_value, max_value)


def cut_rectangle_from_rectangle(area, rectangle):
    assert area <= rectangle_area(rectangle)
    if area == rectangle_area(rectangle):
        chosen_rectangle = rectangle
        rest_rectangle_list = []
    else:
        H = rectangle[3] - rectangle[1]
        W = rectangle[2] - rectangle[0]
        min_chosen_h = math.ceil(area / W)
        max_chosen_h = H
        cnt = 0
        while 1:
            chosen_h = random_select_value(min_chosen_h, max_chosen_h)
            chosen_w = math.floor(area / chosen_h)
            cnt += 1
            if chosen_w > 0:
                break
            if cnt >= 100:
                return rectangle, []
        min_lef_top_x = rectangle[0]
        max_lef_top_x = rectangle[2] - chosen_w
        lef_top_x = random_select_value(min_lef_top_x, max_lef_top_x)

        min_lef_top_y = rectangle[1]
        max_lef_top_y = rectangle[3] - chosen_h
        lef_top_y = random_select_value(min_lef_top_y, max_lef_top_y)
        chosen_rectangle = (lef_top_x, lef_top_y, lef_top_x + chosen_w, lef_top_y + chosen_h)
        rest_rectangle_list = get_rest_rectangle_list(rectangle, chosen_rectangle)
    return chosen_rectangle, rest_rectangle_list


def random_cut_rectangle_from_list(area, rectangle_list):
    # rectangles are (left_top_x, left_top_y, right_bottle_x, right_bottle_y)
    if len(rectangle_list) == 0:
        return None, None
    rectangle_list.sort(key=rectangle_area)
    if area > rectangle_area(rectangle_list[-1]):
        return None, None
    else:
        # find the optimized rectangle
        chosen_rectangle = None
        chosen_index = 0
        for i in range(len(rectangle_list))[::-1]:
            if rectangle_area(rectangle_list[i]) < area:
                chosen_rectangle = rectangle_list[i+1]
                chosen_index = i+1
                break
        if chosen_rectangle is None:
            chosen_rectangle = rectangle_list[0]
        # cut from the chosen_rectangle
        for j in range(chosen_index, len(rectangle_list)):
            cut_rectangle, rest_rectangle_list_append = cut_rectangle_from_rectangle(area, chosen_rectangle)
            if not cut_rectangle is None:
                del(rectangle_list[j])
                return cut_rectangle, rectangle_list+rest_rectangle_list_append
        return None, None


def cut_areas_from_rectangle(area_list, rectangle):
    rectangle_list = [rectangle]
    cut_rectangle_list = []
    for area in area_list:
        chosen_cut, rectangle_list = random_cut_rectangle_from_list(area, rectangle_list)
        if chosen_cut is None:
            return None
        else:
            cut_rectangle_list.append(chosen_cut)
    return cut_rectangle_list


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from tqdm import tqdm

    # 创建一个新的图形和轴
    k = 5
    cnt = 0
    for _ in tqdm(range(100)):
        rand_list = torch.cat([torch.tensor([0]), torch.rand(k - 1),
                               torch.tensor([1])]).sort()[0]
        lam_list = rand_list[1:] - rand_list[:-1]
        area_list = (lam_list.sort()[0] * 224 * 224).int()
        area_list[area_list == 0] += 50
        area_list = area_list.tolist()[::-1][1:]
        rectangle = (0, 0, 224, 224)
        cut_rectangle_list = cut_areas_from_rectangle(area_list, rectangle)

        if cut_rectangle_list is not None:
            cnt += 1
            fig, ax = plt.subplots()
            for rectangle in cut_rectangle_list:
                # 创建一个矩形
                x1, y1, x2, y2 = rectangle
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='g', facecolor='none')
                # 将矩形添加到轴上
                ax.add_patch(rect)
            # 设置轴的范围，反转y轴
            ax.set_xlim(0, 224)
            ax.set_ylim(224, 0)

            # 显示图形
            plt.show()
    print(cnt)



