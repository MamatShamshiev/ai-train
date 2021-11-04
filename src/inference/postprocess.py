import numpy as np
import scipy.ndimage as ndimage


def remove_small_components(
    sem_seg: np.ndarray, max_label: int = 3, thresh=0.001, inplace=True
):
    if inplace is True:
        out = sem_seg
    else:
        out = sem_seg.copy()
    class_boxes = ndimage.find_objects(sem_seg, max_label=max_label)
    for c in range(1, max_label + 1):
        class_box = class_boxes[c - 1]
        if class_box is None:
            continue
        binary_image = sem_seg[class_box] == c
        component_map, num_features = ndimage.label(binary_image)

        drop_inds = []
        for comp_i in range(1, num_features + 1):
            if (component_map == comp_i).sum() / np.prod(sem_seg.shape) < thresh:
                drop_inds.append(comp_i)
        drop_mask = np.isin(component_map, np.array(drop_inds))
        out[class_box][drop_mask] = 0

    return out
