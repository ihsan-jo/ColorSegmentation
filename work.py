import cv2 # computer vision
from skimage.io import imread
import numpy as np # matrix math
from matplotlib import pyplot as plt # Plotting software to help us visualize some things

from sklearn.preprocessing import PolynomialFeatures  # making math equations
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score # solving math equations

import warnings
warnings.filterwarnings("ignore")  # hides some warnings

import streamlit as st

def main():
    # Streamlit input
    st.warning("Image must be in 1x1 aspect ratio. If an error occurs, please reduce the resolution of the image.")
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg"])
    n_clusters = st.number_input("Enter number of clusters", min_value=1, value=2)

    if uploaded_file is not None:
        image_cropped = imread(uploaded_file)
    else:
        st.stop()

    (h,w,c) = image_cropped.shape
    img2D = image_cropped.reshape(h*w,c)
    print(img2D)
    print(img2D.shape)
    # K - means

    from sklearn.cluster import KMeans
    kmeans_model = KMeans(n_clusters) # Ini nanti bisa di-input berapa aja
    cluster_labels = kmeans_model.fit_predict(img2D)
    print(cluster_labels)

    # CLUSTER LABEL

    from collections import Counter
    labels_count = Counter(cluster_labels)
    print(labels_count)
    print("----KOORDINAT RGB----")
    rgb_cols = kmeans_model.cluster_centers_.round(0).astype(int)
    print(rgb_cols)

    from PIL import Image

    img_quant = np.reshape(rgb_cols[cluster_labels],(h,w,c))
    fig, ax = plt.subplots(1,2, figsize=(16,12))
    ax[0].imshow(image_cropped)
    ax[0].set_title('Original Image')
    ax[1].imshow(img_quant[:,:,2])
    ax[1].set_title('Color Quantized Image')

    img_quant = img_quant.astype("uint8")

    import scipy
    scipy.io.savemat('solo25.mat', {'mydata': img_quant})

    st.image(img_quant, caption='Color Quantized Image', use_container_width=True)
    plt.savefig("image.jpg")

    # Convert ndarray gambarnya ke Image

    from PIL import Image as im

    z = img_quant 
    y = z.astype(np.uint8)

    bismillah = im.fromarray(y)

    with bismillah as image:
        color_count = {}
        width, height = bismillah.size
        rgb_image = bismillah.convert('RGB')

        # iterate through each pixel in the image and keep a count per unique color
        for x in range(width):
            for y in range(height):
                rgb = rgb_image.getpixel((x, y))

                if rgb in color_count:
                    color_count[rgb] += 1
                else:
                    color_count[rgb] = 1
                    
        st.write('Jumlah piksel setiap warna:')
        st.write('-' * 30)
        color_index = 1
        for color, count in color_count.items():
            try:
                st.write('{}.) {}: {}'.format(color_index, color, count))
            except ValueError:                
                st.write('{}.) {}: {}'.format(color_index, color, count))
            color_index += 1

    # JANGAN DIUBAH LAGI

    image = image_cropped

    def rgb_to_hex(rgb_color):
        hex_color = "#"
        for i in rgb_color:
            i = int(i)
            hex_color += ("{:02x}".format(i))
        return hex_color

    def prep_image(raw_img):
        modified_img = cv2.resize(raw_img, (1200, 3000), interpolation = cv2.INTER_AREA)
        modified_img = modified_img.reshape(modified_img.shape[0]*modified_img.shape[1], 3)
        return modified_img

    def color_analysis(img):
        clf = kmeans_model
        color_labels = cluster_labels
        center_colors = rgb_cols

        counts = labels_count
        counts = dict(sorted(counts.items()))
        ordered_colors = [center_colors[i] for i in counts.keys()]

        hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
        jumlah = labels_count.values()

        fig, ax = plt.subplots()
        ax.axis("equal")
        pie = ax.pie(jumlah, labels=center_colors, colors=hex_colors)

        ax.set_title("Jumlah Piksel Penyebaran Warna", size=20, weight="bold")
        ax.legend(pie[0], jumlah, bbox_to_anchor=(1, 0.5), loc="center right", fontsize=10, bbox_transform=plt.gcf().transFigure, title='Jumlah Pixel')
        plt.subplots_adjust(left=0.0, bottom=0.1, right=0.85)

        st.pyplot(fig)

    #modified_image = prep_image(image)
    color_analysis(image)

if __name__ == "__main__":
    main()