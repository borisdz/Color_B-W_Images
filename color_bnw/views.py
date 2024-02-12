import base64
import os

import cv2
from cv2 import dnn
import numpy as np
from PIL.Image import Image
from django.core.files.base import ContentFile
from django.http import HttpRequest
from django.shortcuts import render, redirect

from color_bnw.forms import ColorForm
from color_bnw.models import Color, Product


# Create your views here.
def color_image(path):
    PROTO_TXT = r"color_bnw/bnw_assets/colorization_deploy_v2.prototxt"
    POINTS = r"color_bnw/bnw_assets/pts_in_hull.npy"
    MODEL = r"color_bnw/bnw_assets/colorization_release_v2.caffemodel"

    print(os.getcwd())

    print("Load model")
    net = cv2.dnn.readNetFromCaffe(PROTO_TXT, MODEL)
    pts = np.load(POINTS)

    # Load centers for ab channel quantization used for re-balancing
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # Load the input imate
    img = cv2.imread(path.image.path)
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    print("Colorizing the image")
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    ret, frame_buff = cv2.imencode('.jpg', colorized)
    frame_b64 = base64.b64encode(frame_buff)

    colorized = (255 * colorized).astype("uint8")

    # img = Image.fromarray(colorized)
    content = ContentFile(colorized.tobytes())

    # ? Image.frombytes(img, colorized,"raw")

    # cv2.imshow("Original", img)
    # cv2.imshow("Colorized", colorized)
    # cv2.waitKey(0)
    img_product = Product()
    img_product.img_colorized.save(path.image.name, content)
    return img_product.img_colorized.path
    # redirect(success, frame_b64)


def color(request: HttpRequest):
    context = dict()
    context["form"] = ColorForm

    if request.method == "POST":
        form_data = ColorForm(request.POST, request.FILES)
        if form_data.is_valid():
            received_img: Color = form_data.save(commit=False)
            received_img.image = form_data.cleaned_data["image"]
            received_img.save()

            colorized_img = color_image(received_img)

            color_image_2(received_img)

            return redirect(f"?image={colorized_img}")

    return render(request, "color_bnw.html", context=context)


def success(request: HttpRequest, frame: bytes):
    frame_64 = frame
    return render(request, "colored_img.html", context={"frame": frame_64})


def color_image_2(path):
    proto_file = r"color_bnw/bnw_assets/colorization_deploy_v2.prototxt"
    hull_pts = r"color_bnw/bnw_assets/pts_in_hull.npy"
    model_file = r"color_bnw/bnw_assets/colorization_release_v2.caffemodel"

    # Read model params.
    net = dnn.readNetFromCaffe(proto_file,model_file)
    kernel = np.load(hull_pts)

    # Reading and processing image
    img = cv2.imread(path.image.path)
    scaled = img.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # add cluster centers as 1x1 convolutions to the model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = kernel.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # resize the image for the network
    resized = cv2.resize(lab_img, (224, 224))
    # split the L channel
    L = cv2.split(resized)[0]
    # mean subtraction
    L -= 50

    # predicting the ab channels from the input L channel
    net.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    # resize the predicted 'ab' volume to the same dimension as input img
    ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))

    # Take the L channel from the image
    L = cv2.split(lab_img)[0]
    # Join the L channel with predicted ab channel
    colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)

    # Convert the img from Lab to BGR
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # change the img to 0-255 range and convert it from float32 to int
    colorized = (255 * colorized).astype("uint8")

    # resize the image and show together
    img = cv2.resize(img, (640, 640))
    colorized = cv2.resize(colorized, (640, 640))

    result = cv2.hconcat([img, colorized])

    cv2.imshow("Grayscale -> Color", result)

    cv2.waitKey()
