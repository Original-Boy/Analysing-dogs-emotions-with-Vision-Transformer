import imgaug.augmenters as iaa
import cv2
import os
seq=[]
seq.append(iaa.Sequential([iaa.Fliplr(0.5)]))
seq.append(iaa.Sequential([iaa.Affine(rotate=(-20, 20))]))
seq.append(iaa.Sequential([iaa.Affine(scale=(0.1, 0.4),translate_px={"x": (0, 100), "y": (0, 70)})]))
seq.append(iaa.Sequential([iaa.Affine(scale=(1, 2),translate_px={"x": (0, 100), "y": (0, 70)})]))
seq.append(iaa.Sequential([iaa.Dropout(p=(0, 0.2))]))
seq.append(iaa.Sequential([iaa.AddToHueAndSaturation((-50, 50), per_channel=True)]))
seq.append(iaa.Sequential([iaa.GaussianBlur(sigma=(0.0, 4.0))]))
seq.append(iaa.Sequential([iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05))]))


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

imgaug = []

imglist=load_images_from_folder('D:\\all')
for i in range(0, 8):
    imgaug.append(seq[i].augment_images(imglist))
    print(i,len(imglist),len(imgaug))

type=['flip','rotate','translate','crop','noise','color-space','gaussian','weather']
n=0;
for i in imgaug:
    num=0
    for k in imgaug[n]:
        cv2.imwrite( 'output\\all\\'+str(n) +type[n] +str(num)+ '.jpg', k)
        num += 1
    n+=1