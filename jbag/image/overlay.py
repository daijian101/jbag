from typing import Union

import SimpleITK as sitk


def overlay(image, label_map, colors: Union[list, tuple] = None, opacity=0.5):
    image = sitk.GetImageFromArray(image)
    label_map = sitk.GetImageFromArray(label_map)
    if colors:
        if len(colors) == 1:
            itk_color_map = list(colors[0])
        else:
            itk_color_map = list(colors[-1])
            for i in colors[:-1]:
                itk_color_map += list(i)
        overlaid_image = sitk.LabelOverlay(image, label_map, opacity=opacity, colormap=itk_color_map)
    else:
        overlaid_image = sitk.LabelOverlay(image, label_map, opacity=opacity)
    overlaid_image = sitk.GetArrayFromImage(overlaid_image)
    return overlaid_image
