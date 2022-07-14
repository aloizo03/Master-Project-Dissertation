import cv2


def transform(image, size, min_norm=0, max_norm=1., transpose=(2, 0, 1)):
    img_resize = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    img_normalize = cv2.normalize(img_resize, None, min_norm, max_norm, cv2.NORM_MINMAX)
    img_normalize = img_normalize.transpose(transpose)
    return img_normalize
