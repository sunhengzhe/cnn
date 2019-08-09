import numpy as np

class MaxPoolLayer:
    def iterate_regions(self, image):
        n, w, h = image.shape

        for i in range(w // 2):
            for j in range(h // 2):
                block = image[i*2:i*2+2, j*2:j*2+2]
                yield block, i, j

    def forward(self, input):
        self.last_input = input

        w, h, n = input.shape

        out = np.zeros((w // 2, h // 2, n))

        for block, i, j in self.iterate_regions(input):
            out[i, j] = np.amax(block, axis = (0, 1))

        return out

    def backprop(self, d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
          h, w, f = im_region.shape
          amax = np.amax(im_region, axis=(0, 1))

          for i2 in range(h):
            for j2 in range(w):
              for f2 in range(f):
                # If this pixel was the max value, copy the gradient to it.
                if im_region[i2, j2, f2] == amax[f2]:
                  d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input
