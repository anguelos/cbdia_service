import torch
from torch.nn import Parameter
import logging
import os
import sys
from skimage.transform import resize
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import tqdm
import cv2
import io

class HomographyAugmentation(object):
    '''
    Class for creating homography augmentation transformations in the pytorch
    framework.
    '''

    def __init__(self, random_limits=(0.9, 1.1)):
        '''
        Constructor
        '''
        self.random_limits = random_limits

    def __call__(self, img):
        '''
        Creates an augmentation by computing a homography from three
        points in the image to three randomly generated points
        '''
        y, x = img.shape[:2]
        fx = float(x)
        fy = float(y)
        src_point = np.float32([[fx/2, fy/3,],
                                [2*fx/3, 2*fy/3],
                                [fx/3, 2*fy/3]])
        random_shift = (np.random.rand(3,2) - 0.5) * 2 * (self.random_limits[1]-self.random_limits[0])/2 + np.mean(self.random_limits)
        dst_point = src_point * random_shift.astype(np.float32)
        transform = cv2.getAffineTransform(src_point, dst_point)
        #border_value = 0
        if img.ndim == 3:
            border_value = np.median(np.reshape(img, (img.shape[0]*img.shape[1], -1)), axis=0)
        else:
            border_value = np.median(img)
        warped_img = cv2.warpAffine(img, transform, dsize=(x,y), borderValue=float(border_value))
        return warped_img

class LineListIO(object):
    '''
    Helper class for reading/writing text files into lists.
    The elements of the list are the lines in the text file.
    '''

    @staticmethod
    def read_list(filepath, encoding='ascii'):
        if not os.path.exists(filepath):
            raise ValueError('File for reading list does NOT exist: ' + filepath)

        linelist = []
        if encoding == 'ascii':
            transform = lambda line: line.encode()
        else:
            transform = lambda line: line

        with io.open(filepath, encoding=encoding) as stream:
            for line in stream:
                line = transform(line.strip())
                if line != '':
                    linelist.append(line)
        return linelist

    @staticmethod
    def write_list(file_path, line_list, encoding='ascii',
                   append=False, verbose=False):
        '''
        Writes a list into the given file object

        file_path: the file path that will be written to
        line_list: the list of strings that will be written
        '''
        mode = 'w'
        if append:
            mode = 'a'

        with io.open(file_path, mode, encoding=encoding) as f:
            if verbose:
                line_list = tqdm.tqdm(line_list)

            for l in line_list:
                f.write(unicode(l) + '\n')


def average_precision(ret_vec_relevance, gt_relevance_num=None):
    '''
    Computes the average precision from a list of relevance items

    Params:
        ret_vec_relevance: A 1-D numpy array containing ground truth (gt)
            relevance values
        gt_relevance_num: Number of relevant items in the data set
            (with respect to the ground truth)
            If None, the average precision is calculated wrt the number of
            relevant items in the retrieval list (ret_vec_relevance)
    Returns:
        The average precision for the given relevance vector.
    '''
    if ret_vec_relevance.ndim != 1:
        raise ValueError('Invalid ret_vec_relevance shape')

    ret_vec_cumsum = np.cumsum(ret_vec_relevance, dtype=float)
    ret_vec_range = np.arange(1, ret_vec_relevance.size + 1)
    ret_vec_precision = ret_vec_cumsum / ret_vec_range

    if gt_relevance_num is None:
        n_relevance = ret_vec_relevance.sum()
    else:
        n_relevance = gt_relevance_num

    if n_relevance > 0:
        ret_vec_ap = (ret_vec_precision * ret_vec_relevance).sum() / n_relevance
    else:
        ret_vec_ap = 0.0
    return ret_vec_ap


def map_from_query_test_feature_matrices(query_features, test_features,
                                         query_labels, test_labels,
                                         metric,
                                         drop_first=False):
    '''
    Compute the mAP for a given list of queries and test instances
    Each query is used to rank the test samples
    :param query_features: (2D ndarray)
        feature representation of the queries
    :param test_features: (2D ndarray)
        feature representation of the test instances
    :param query_labels: (1D ndarray or list)
        the labels corresponding to the queries (either numeric or characters)
    :param test_labels: (1D ndarray or list)
        the labels corresponding to the test instances (either numeric or characters)
    :param metric: (string)
        the metric to be used in calculating the mAP
    :param drop_first: (bool)
        whether to drop the first retrieval result or not
    '''
    # some argument error checking
    if query_features.shape[1] != test_features.shape[1]:
        raise ValueError('Shape mismatch')
    if query_features.shape[0] != len(query_labels):
        raise ValueError('The number of query feature vectors and query labels does not match')
    if test_features.shape[0] != len(test_labels):
        raise ValueError('The number of test feature vectors and test labels does not match')

    # compute the nearest neighbors
    dist_mat = cdist(XA=query_features, XB=test_features, metric=metric)
    retrieval_indices = np.argsort(dist_mat, axis=1)

    # create the retrieval matrix
    retr_mat = np.tile(test_labels, (len(query_labels), 1))
    row_selector = np.transpose(np.tile(np.arange(len(query_labels)), (len(test_labels), 1)))
    retr_mat = retr_mat[row_selector, retrieval_indices]

    # create the relevance matrix
    relevance_matrix = retr_mat == np.atleast_2d(query_labels).T
    if drop_first:
        relevance_matrix = relevance_matrix[:, 1:]

    # calculate mAP and APs
    avg_precs = np.array([average_precision(row) for row in relevance_matrix], ndmin=2).flatten()
    mean_ap = np.mean(avg_precs)
    return mean_ap, avg_precs


def map_from_feature_matrix(features, labels, metric, drop_first):
    '''
    Computes mAP and APs from a given matrix of feature vectors
    Each sample is used as a query once and all the other samples are
    used for testing. The user can specify whether he wants to include
    the query in the test results as well or not.
    :param features:(2D ndarray)
        the feature representation from which to compute the mAP
    :param labels: (1D ndarray or list)
        the labels corresponding to the features (either numeric or characters)
    :param metric: (string)
        the metric to be used in calculating the mAP
    :param drop_first: (bool)
        whether to drop the first retrieval result or not
    '''
    # argument error checks
    if features.shape[0] != len(labels):
        raise ValueError('The number of feature vectors and number of labels must match')
    # compute the pairwise distances from the features
    dist_mat = squareform(pdist(X=features, metric=metric))
    np.fill_diagonal(dist_mat, -1)  # make sure identical indices are sorted to the front
    inds = np.argsort(dist_mat, axis=1)
    retr_mat = np.tile(labels, (features.shape[0], 1))

    # compute two matrices for selecting rows and columns
    # from the label matrix
    # -> advanced indexing
    row_selector = np.transpose(np.tile(np.arange(features.shape[0]), (features.shape[0], 1)))
    retr_mat = retr_mat[row_selector, inds]

    # create the relevance matrix
    rel_matrix = retr_mat == np.atleast_2d(labels).T
    if drop_first:
        rel_matrix = rel_matrix[:, 1:]

    # calculate mAP and APs
    avg_precs = np.array([average_precision(row) for row in rel_matrix])
    mean_ap = np.mean(avg_precs)
    return mean_ap, avg_precs


def check_size(img, min_image_width_height, fixed_image_size=None):
    """

    checks if the image accords to the minimum and maximum size requirements
    or fixed image size and resizes if not

    :param img: the image to be checked
    :param min_image_width_height: the minimum image size
    :param fixed_image_size:
    """

    if fixed_image_size is not None:
        if len(fixed_image_size) != 2:
            raise ValueError('The requested fixed image size is invalid!')
        new_img = resize(image=img, output_shape=fixed_image_size[::-1])
        new_img = new_img.astype(np.float32)
        return new_img
    elif np.amin(img.shape[:2]) < min_image_width_height:
        if np.amin(img.shape[:2]) == 0:
            return None
        scale = float(min_image_width_height + 1) / float(np.amin(img.shape[:2]))
        new_shape = (int(scale * img.shape[0]), int(scale * img.shape[1]))
        new_img = resize(image=img, output_shape=new_shape)
        new_img = new_img.astype(np.float32)
        return new_img
    else:
        return img


class AverageMeter(object):
  """
  Computes and stores the average and current value
  """
  def __init__(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val*n
    self.count += n
    self.avg = self.sum/self.count


class Statistics(object):
  def __init__(self, names):
    self.names = names
    self.meters = {}
    for name in names:
      self.meters.update({name: AverageMeter()})

  def update(self, n, **kwargs):
    info = ''
    for key in kwargs:
      self.meters[key].update(kwargs[key], n)
      info += '{key}={loss.val:.4f}, avg {key}={loss.avg:.4f}, '.format(key=key, loss=self.meters[key])
    return info[:-2]

  def summary(self):
    info = ''
    for name in self.names:
      info += 'avg {key}={loss:.4f}, '.format(key=name, loss=self.meters[name].avg)
    return info[:-2]


class Logger(object):
  def __init__(self, path):
    self.logger = logging.getLogger()
    self.logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    fh = logging.FileHandler(os.path.join(path, 'debug.log'))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    self.logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    self.logger.addHandler(ch)

  def log(self, info):
    self.logger.info(info)

def my_torch_save(model, filename):

    # cpu save
    #if next(model.parameters()).is_cuda:
    #    model = model.cpu()

    model_parameters = {name : param.data.cpu() for name, param in model.named_parameters()}
    torch.save(model_parameters, filename)


def my_torch_load(model, filename, use_list=None):
    #use_cuda=next(model.parameters()).is_cuda
    model_parameters = torch.load(filename)
    own_state = model.state_dict()

    for name in model_parameters.keys():
        if use_list is not None:
            if name not in use_list:
                continue
        if name in own_state:
            #print name
            param = model_parameters[name]
            if isinstance(param, Parameter):
                param = param.data

            if own_state[name].shape[:] != param.shape[:]:
                continue
            own_state[name].copy_(param)
