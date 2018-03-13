# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import numpy as np
import scipy.misc
import os

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x

# use tensorboardX
try:
    from tensorboardX import SummaryWriter
    import torch
    import torchvision.utils as vutils
    use_tensorboardX = True
except ImportError:
    use_tensorboardX = False


class TensorboardLogger_PureTF(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer_train = tf.summary.FileWriter(os.path.join(log_dir, "train"))
        self.writer_valid = tf.summary.FileWriter(os.path.join(log_dir, "valid"))

    def scalar_summary(self, tag, value, step, is_train=True):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        if is_train:
            self.writer_train.add_summary(summary, step)
        else:
            self.writer_valid.add_summary(summary, step)

    def image_summary(self, tag, images, step, is_train=True):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)

        if is_train:
            self.writer_train.add_summary(summary, step)
        else:
            self.writer_valid.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000, is_train=True):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(int(c))

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        if is_train:
            self.writer_train.add_summary(summary, step)
            self.writer_train.flush()
        else:
            self.writer_valid.add_summary(summary, step)
            self.writer_valid.flush()


class TensorboardLogger_X(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer_train = SummaryWriter(os.path.join(log_dir, "train"))
        self.writer_valid = SummaryWriter(os.path.join(log_dir, "valid"))

    def scalar_summary(self, tag, value, step, is_train=True):
        """Log a scalar variable."""
        if is_train:
            self.writer_train.add_scalar(tag, value, step)
        else:
            self.writer_valid.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step, is_train=True):
        """Log a list of images."""
        # Input: images - numpy.array [N, 3, H, W]
        images = torch.from_numpy(images)  # the bridge to TensorboardLogger_PureTF
        images = vutils.make_grid(images, normalize=True, scale_each=True)
        if is_train:
            self.writer_train.add_summary(tag, images, step)
        else:
            self.writer_valid.add_summary(tag, images, step)

    def histo_summary(self, tag, values, step, bins='auto', is_train=True):
        """Log a histogram of the tensor of values."""
        # Input: values - numpy.array
        if is_train:
            self.writer_train.add_histogram(tag, values, step, bins=bins)
        else:
            self.writer_valid.add_histogram(tag, values, step, bins=bins)

    def embedding_summary(self, tag, embedding_mat, labels=[], images=None, step=None, is_train=True):
        """Plot embedding data after a mini-batch."""
        '''
        Inputs:
            embedding_mat - torch.Tensor [N, D]
            images - torch.Tensor [N, 3, H, W]
        '''
        if is_train:
            self.writer_train.add_embedding(mat=embedding_mat, metadata=labels, label_img=images, global_step=step, tag=tag)
        else:
            self.writer_valid.add_embedding(mat=embedding_mat, metadata=labels, label_img=images, global_step=step, tag=tag)


if not use_tensorboardX:
    print('INFO:', 'enjoy embedding visualization with `pip install tensorboardX`')

    class TensorboardLogger(TensorboardLogger_PureTF):

        def __init__(self, log_dir):
            super(TensorboardLogger_PureTF, self).__init__(log_dir)
else:
    class TensorboardLogger(TensorboardLogger_X):

        def __init__(self, log_dir):
            super(TensorboardLogger, self).__init__(log_dir)
