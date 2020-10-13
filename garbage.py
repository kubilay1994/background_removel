mask = decode_segmap(arr)


fg = cv2.imread("images/cat.jpg")
bg = cv2.imread("images/bg.jpg")

bg = cv2.resize(
    bg, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA
).astype(float)

fg = fg.astype(float)

mask = np.dot(mask[..., :3], [0.299, 0.587, 0.114])
_, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
mask = mask[..., np.newaxis]
cv2.imshow("mask", mask)

mask = mask / 255

# fg = cv2.multiply(mask, fg)
fg = mask * fg
bg = (1.0 - mask) * bg
# bg = cv2.multiply(1.0 - mask, bg)

# final = cv2.add(fg, bg).astype(np.uint8)
final = (fg + bg).astype(np.uint8)

mask = (mask * 255).astype(np.uint8)
cv2.imshow("final", final)
cv2.waitKey(0)

# Image.fromarray(arr).resize(img.size).show()


def decode_segmap(image, nc=21):
    label_colors = np.array(
        [
            (0, 0, 0),  # 0=background
            # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
            (128, 0, 0),
            (0, 128, 0),
            (255, 255, 255),
            (0, 0, 128),
            (128, 0, 128),
            # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
            (0, 128, 128),
            (128, 128, 128),
            (80, 0, 0),
            (192, 0, 0),
            (64, 128, 0),
            # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
            (192, 128, 0),
            (64, 0, 128),
            (192, 0, 128),
            (64, 128, 128),
            (255, 255, 255),
            # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
            (0, 64, 0),
            (128, 64, 0),
            (0, 192, 0),
            (128, 192, 0),
            (0, 64, 128),
        ]
    )
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
        rgb = np.stack([r, g, b], axis=2)
    return rgb