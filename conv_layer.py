import numpy as np

class ConvLayer:
    def __init__(self, num_of_filters):
        self.num_of_filters = num_of_filters
        self.filters = np.random.randn(num_of_filters, 3, 3) / 9

    def iterate_regions(self, image):
        w, h = image.shape

        for i in range(w - 2):
            for j in range(h - 2):
                block = image[i:i+3, j:j+3]
                yield block, i, j

    def forward(self, img):
        self.last_input = img
        w, h = img.shape
        out = np.zeros((w - 2, h - 2, self.num_of_filters))

        for img_region, i, j in self.iterate_regions(self.last_input):
            out[i, j] = np.sum(img_region * self.filters, axis=(1, 2))

        return out


    def backprop(self, d_L_d_out, learn_rate):
        d_L_d_filter = np.zeros(self.filters.shape)

        for img_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_of_filters):
                d_L_d_filter[f] += d_L_d_out[i, j, f] * img_region

        self.filters -= learn_rate * d_L_d_filter

        return None

