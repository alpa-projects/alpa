import numpy as np


def show_images(images, cols=1, titles=None, show=True):
    """Display a list of images in a single figure with matplotlib.
    https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    from matplotlib import pyplot as plt
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 3 and image.shape[2] == 1:
            image = np.squeeze(image, axis=2)
            gray = True
        if image.ndim == 2:
            gray = True
        else:
            gray = False

        if gray:
            plt.gray()

        if image.dtype == np.uint8:
            a.imshow(image, cmap='gray' if gray else 'jet', vmin=0, vmax=255)
        else:
            a.imshow(image, cmap='gray' if gray else 'jet')

        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    if show:
        plt.show()
