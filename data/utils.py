from PIL import Image

def resize_img(img_path, out_path, size=(224, 224), is_mask=False):
    img = Image.open(img_path)
    if is_mask:
        img = img.resize(size, resample=Image.NEAREST)
    else:
        img = img.resize(size, resample=Image.BILINEAR)
    img.save(out_path)