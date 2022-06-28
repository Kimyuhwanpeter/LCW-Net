# -*- coding:utf-8 -*-
from charset_normalizer import from_bytes
from base_UNET_v2 import *
from base_UNET import *
# from modified_deeplab_V3 import *
from paper5th_model import *
from PFB_measurement import Measurement
from random import shuffle, random
from tensorflow.keras import backend as K
# from keras_flops import get_flops
# from model_profiler import model_profiler

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 512,

                           "train_txt_path": "/yuwhan/yuwhan/Dataset/Segmentation/CWFID/train_fix.txt",

                           "val_txt_path": "/yuwhan/yuwhan/Dataset/Segmentation/CWFID/val.txt",

                           "test_txt_path": "/yuwhan/yuwhan/Dataset/Segmentation/CWFID/test.txt",
                           
                           "label_path": "/yuwhan/yuwhan/Dataset/Segmentation/CWFID/aug_train_masks/",
                           
                           "image_path": "/yuwhan/yuwhan/Dataset/Segmentation/CWFID/low_light2/",
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V3_ablation/without_loss/CWFID_v2/checkpoint/70",
                           
                           "lr": 0.0001,

                           "min_lr": 1e-7,
                           
                           "epochs": 400,

                           "total_classes": 3,

                           "ignore_label": 0,

                           "batch_size": 4,

                           "sample_images": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V3/CWFID/sample_images",

                           "save_checkpoint": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V3/CWFID/checkpoint",

                           "save_print": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V3/CWFID/train_out.txt",

                           "train_loss_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V3/CWFID/train_loss.txt",

                           "train_acc_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V3/CWFID/train_acc.txt",

                           "val_loss_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V3/CWFID/val_loss.txt",

                           "val_acc_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V3/CWFID/val_acc.txt",

                           "test_images": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V3_ablation/without_loss/CWFID_v2/test_images",

                           "train": True})


optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
optim2 = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
color_map = np.array([[255, 0, 0], [0, 0, 255], [0,0,0]], dtype=np.uint8)

def tr_func(image_list, label_list):

    h = tf.random.uniform([1], 1e-2, 30)
    h = tf.cast(tf.math.ceil(h[0]), tf.int32)
    w = tf.random.uniform([1], 1e-2, 30)
    w = tf.cast(tf.math.ceil(w[0]), tf.int32)

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.cast(img, tf.float32)
    # img = tf.image.random_brightness(img, max_delta=50.) 
    # img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    # img = tf.image.random_hue(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=1.0, upper=2.0)
    img = tf.clip_by_value(img, 0, 255)
    # img = tf.image.random_crop(img, [FLAGS.img_size, FLAGS.img_size, 3], seed=123)
    no_img = img
    # img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68]) # 평균값 보정
    img = img / 255.

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # lab = tf.image.random_crop(lab, [FLAGS.img_size, FLAGS.img_size, 1], seed=123)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    if random() > 0.5:
        img = tf.image.flip_left_right(img)
        lab = tf.image.flip_left_right(lab)

    return img, no_img, lab

def test_func(image_list, label_list):

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.clip_by_value(img, 0, 255)
    # img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68]) # 평균값 보정
    img = img / 255.

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    return img, lab

def test_func2(image_list, label_list):

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.clip_by_value(img, 0, 255)
    temp_img = img
    # img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68]) # 평균값 보정
    img = img / 255.

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    return img, temp_img, lab

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def true_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - tf.math.divide(numerator, denominator)

def false_dice_loss(y_true, y_pred):
    y_true = 1 - tf.cast(y_true, tf.float32)
    y_pred = 1 - tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - tf.math.divide(numerator, denominator)

def modified_dice_loss_object(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    # y_pred = tf.math.sigmoid(y_pred)
    numerator = (y_true * (1 - tf.math.sigmoid(y_pred)) * tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred))
    denominator = (y_true * tf.math.sigmoid(y_pred) * 0.9 * tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred) + 1)
    loss = tf.math.divide(numerator, denominator)
    loss = tf.reduce_mean(loss)

    return loss

def modified_dice_loss_nonobject(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    # y_pred = tf.math.sigmoid(y_pred)
    numerator = ((1  - y_true) * tf.math.sigmoid(y_pred) * tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred))
    denominator = ((1  - y_true) * (1 - tf.math.sigmoid(y_pred)) * 0.9 * tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred) + 1)
    loss = tf.math.divide(numerator, denominator)
    loss = tf.reduce_mean(loss)

    return loss

def two_region_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2*(tf.reduce_sum(y_true*y_pred) + tf.reduce_sum((1 - y_true)*(1 - y_pred)))
    denominator = tf.reduce_sum(y_true + y_pred) + tf.reduce_sum(2 - y_true - y_pred)

    return 1 - tf.math.divide(numerator, denominator)

def two_region_dice_loss_w_onehot(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    numerator = 2*(tf.reduce_sum(y_true*y_pred) + tf.reduce_sum((1 - y_true)*(1 - y_pred)))
    denominator = tf.reduce_sum(y_true + y_pred) + tf.reduce_sum(2 - y_true - y_pred)

    return 1 - tf.math.divide(numerator, denominator)

def categorical_focal_loss(alpha, gamma=2.):
    """
    Softmax version of focal loss.
    When there is a skew between different categories/labels in your data set, you can try to apply this function as a
    loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy. Alpha is used to specify the weight of different
      categories/labels, the size of the array needs to be consistent with the number of classes.
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=[[.25, .25, .25]], gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    alpha = np.array(alpha, dtype=np.float32)
    alpha = np.reshape(alpha, [1, 2])

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.experimental.numpy.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return tf.keras.backend.mean(tf.keras.backend.sum(loss, axis=-1))
        # return (tf.keras.backend.sum(loss, axis=-1))

    return categorical_focal_loss_fixed

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -tf.keras.backend.mean((alpha * tf.math.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) \
               + ((1 - alpha) * tf.math.pow(pt_0, gamma) * tf.math.log(1. - pt_0)))
        # return -tf.keras.backend.sum(alpha * tf.math.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) \
        #        -tf.keras.backend.sum((1 - alpha) * tf.math.pow(pt_0, gamma) * tf.math.log(1. - pt_0))

    return binary_focal_loss_fixed

def Combo_loss_dice_focal(y_true, y_pred, class_imbal_labels_buf, crop_buf, weed_buf):

    # ce_w values smaller than 0.5 penalize false positives more. ce_w values larger than 0.5 panalize false negative more
    # - (ALPHA * ((targets * K.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * K.log(1.0 - inputs))) )    
    # (crop-0 (negative), weed-1 (positive))
    # for crop and weed--> if crop > weed; FN
    # for crop and weed--> if crop < weed; FP
    # for crop and weed--> if crop = weed; ce_w = 0.5

    # combo_alpha; controls the amount of Dice term contribution (best combo_alpha is 0.5 (from paper))
    combo_alpha = 0.5
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.nn.sigmoid(y_pred)
    numerator = 2*(tf.keras.backend.sum(y_true*y_pred) + tf.keras.backend.sum((1 - y_true)*(1 - y_pred)))       # crop dice ; weed dice devide!!!!!!!!!!!!!!
    denominator = tf.keras.backend.sum(y_true + y_pred) + tf.keras.backend.sum(2 - y_true - y_pred)
    dice = 1 - (numerator / denominator)

    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
    if class_imbal_labels_buf[0] < class_imbal_labels_buf[1]:
        ce_w = weed_buf[1]
        alpha = weed_buf[1]
        distribution = -(ce_w * (y_true * alpha * tf.math.pow(1. - y_pred, 2.) * tf.math.log(y_pred)) + ((1 - ce_w) * (1 - y_true) * (1 - alpha) * tf.math.pow(y_pred, 2.) * tf.math.log(1 - y_pred)))
        distribution = tf.keras.backend.mean(distribution, axis=-1)
        loss = (combo_alpha * distribution) + ((1 - combo_alpha) * dice)
    elif class_imbal_labels_buf[0] > class_imbal_labels_buf[1]:
        ce_w = crop_buf[1]
        alpha = crop_buf[1]
        distribution = -(ce_w * (y_true * alpha * tf.math.pow(1. - y_pred, 2.) * tf.math.log(y_pred)) + ((1 - ce_w) * (1 - y_true) * (1 - alpha) * tf.math.pow(y_pred, 2.) * tf.math.log(1 - y_pred)))
        distribution = tf.keras.backend.mean(distribution, axis=-1)
        loss = (combo_alpha * distribution) + ((1 - combo_alpha) * dice)
    else:
        ce_w = 0.5
        alpha = 0.5
        distribution = -(ce_w * (y_true * alpha * tf.math.pow(1. - y_pred, 2.) * tf.math.log(y_pred)) + ((1 - ce_w) * (1 - y_true) * (1 - alpha) * tf.math.pow(y_pred, 2.) * tf.math.log(1 - y_pred)))
        distribution = tf.keras.backend.mean(distribution, axis=-1)
        loss = (combo_alpha * distribution) + ((1 - combo_alpha) * dice)

    # print(distribution)
    # print(dice)
    return loss

def cal_loss(model, model2, images, labels, objectiness, class_imbal_labels_buf, object_buf, crop_buf, weed_buf):

    with tf.GradientTape() as tape: # channel ==> 1

        batch_labels = tf.reshape(labels, [-1,])
        raw_logits = run_model(model, images, True)
        label_objectiness = tf.cast(tf.reshape(objectiness, [-1,]), tf.float32)
        logit_objectiness = tf.reshape(raw_logits, [-1,], tf.float32)

        total_loss = true_dice_loss(label_objectiness, logit_objectiness)

    grads = tape.gradient(total_loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    with tf.GradientTape() as tape2: # channel ==> 2

        batch_labels = tf.reshape(labels, [-1,])
        logits = run_model(model2, images * tf.nn.sigmoid(raw_logits), True)
        logits = tf.reshape(logits, [-1, FLAGS.total_classes-1])
        objectiness = np.where(batch_labels == 2, 0, 1)
        object_loss = true_dice_loss(objectiness, logits[:, 1])
        
        crop_weed_indices = tf.squeeze(tf.where(tf.not_equal(batch_labels, 2)), -1).numpy()
        crop_weed_labels = tf.gather(batch_labels, crop_weed_indices)
        crop_weed_logits = tf.gather(logits[:, 0], crop_weed_indices)

        if class_imbal_labels_buf[0] < class_imbal_labels_buf[1]:
            crop_weed_distri_loss = binary_focal_loss(alpha=weed_buf[1])(crop_weed_labels, tf.nn.sigmoid(crop_weed_logits))
            crop_weed_dice_loss = true_dice_loss(crop_weed_labels, crop_weed_logits)

            only_weed_output = tf.where(batch_labels == 1, tf.nn.sigmoid(logits[:, 0]), logits[:, 1])   # without this == > ablation 2
            only_weed_indices = tf.where(batch_labels == 1).numpy() # without this == > ablation 2
            only_weed_logits = tf.gather(only_weed_output, only_weed_indices)   # without this == > ablation 2
            only_one_loss = tf.reduce_mean( -tf.math.log(only_weed_logits + tf.keras.backend.epsilon()) )   # without this == > ablation 2
        else:
            crop_weed_distri_loss = binary_focal_loss(alpha=weed_buf[0])(crop_weed_labels, tf.nn.sigmoid(crop_weed_logits))
            crop_weed_dice_loss = false_dice_loss(crop_weed_labels, crop_weed_logits)

            only_crop_output = tf.where(batch_labels == 0, 1. - tf.nn.sigmoid(logits[:, 0]), logits[:, 1])  # without this == > ablation 2
            only_crop_indices = tf.where(batch_labels == 0).numpy() # without this == > ablation 2
            only_crop_logits = tf.gather(only_crop_output, only_crop_indices)   # without this == > ablation 2
            only_one_loss = tf.reduce_mean( -tf.math.log(only_crop_logits + tf.keras.backend.epsilon()) )   # without this == > ablation 2

        # crop_weed_combo_loss = Combo_loss_dice_focal(crop_weed_labels, crop_weed_logits, class_imbal_labels_buf, crop_buf, weed_buf)

        ##############################################################################################################################
        only_background_output  = tf.where(objectiness == 0, 1. - tf.nn.sigmoid(logits[:, 1]), logits[:, 0]).numpy()    # without this == > ablation 2
        only_background_indices = tf.where(objectiness == 0).numpy()    # without this == > ablation 2
        only_background_logits = tf.gather(only_background_output, only_background_indices) # without this == > ablation 2
        only_background_loss = tf.reduce_mean( -tf.math.log(only_background_logits + tf.keras.backend.epsilon()) )  # without this == > ablation 2
        ##############################################################################################################################

        total_loss = object_loss + crop_weed_dice_loss + crop_weed_distri_loss + only_background_loss + only_one_loss
        # total_loss = object_loss + crop_weed_dice_loss + crop_weed_distri_loss
        
        
    grads = tape2.gradient(total_loss, model2.trainable_variables)
    optim2.apply_gradients(zip(grads, model2.trainable_variables))

    return total_loss

def val_cal_loss(model, model2, images, labels, objectiness, class_imbal_labels_buf, object_buf, crop_buf, weed_buf):

    with tf.GradientTape() as tape: # channel ==> 1

        batch_labels = tf.reshape(labels, [-1,])
        raw_logits = run_model(model, images, False)
        label_objectiness = tf.cast(tf.reshape(objectiness, [-1,]), tf.float32)
        logit_objectiness = tf.reshape(raw_logits, [-1,], tf.float32)

        total_loss = true_dice_loss(label_objectiness, logit_objectiness)

    with tf.GradientTape() as tape2: # channel ==> 2

        batch_labels = tf.reshape(labels, [-1,])
        logits = run_model(model2, images * tf.nn.sigmoid(raw_logits), False)
        logits_ = logits
        logits = tf.reshape(logits, [-1, FLAGS.total_classes-1])
        objectiness = np.where(batch_labels == 2, 0, 1)
        object_loss = true_dice_loss(objectiness, logits[:, 1])
        
        crop_weed_indices = tf.squeeze(tf.where(tf.not_equal(batch_labels, 2)), -1).numpy()
        crop_weed_labels = tf.gather(batch_labels, crop_weed_indices)
        crop_weed_logits = tf.gather(logits[:, 0], crop_weed_indices)

        if class_imbal_labels_buf[0] < class_imbal_labels_buf[1]:
            crop_weed_distri_loss = binary_focal_loss(alpha=weed_buf[1])(crop_weed_labels, tf.nn.sigmoid(crop_weed_logits))
            crop_weed_dice_loss = true_dice_loss(crop_weed_labels, crop_weed_logits)

            only_weed_output = tf.where(batch_labels == 1, tf.nn.sigmoid(logits[:, 0]), logits[:, 1])   # without this == > ablation 2
            only_weed_indices = tf.where(batch_labels == 1).numpy() # without this == > ablation 2
            only_weed_logits = tf.gather(only_weed_output, only_weed_indices)   # without this == > ablation 2
            only_one_loss = tf.reduce_mean( -tf.math.log(only_weed_logits + tf.keras.backend.epsilon()) )   # without this == > ablation 2
        else:
            crop_weed_distri_loss = binary_focal_loss(alpha=weed_buf[0])(crop_weed_labels, tf.nn.sigmoid(crop_weed_logits))
            crop_weed_dice_loss = false_dice_loss(crop_weed_labels, crop_weed_logits)

            only_crop_output = tf.where(batch_labels == 0, 1. - tf.nn.sigmoid(logits[:, 0]), logits[:, 1])  # without this == > ablation 2
            only_crop_indices = tf.where(batch_labels == 0).numpy() # without this == > ablation 2
            only_crop_logits = tf.gather(only_crop_output, only_crop_indices)   # without this == > ablation 2
            only_one_loss = tf.reduce_mean( -tf.math.log(only_crop_logits + tf.keras.backend.epsilon()) )   # without this == > ablation 2

        # crop_weed_combo_loss = Combo_loss_dice_focal(crop_weed_labels, crop_weed_logits, class_imbal_labels_buf, crop_buf, weed_buf)

        ##############################################################################################################################
        only_background_output  = tf.where(objectiness == 0, 1. - tf.nn.sigmoid(logits[:, 1]), logits[:, 0]).numpy()    # without this == > ablation 2
        only_background_indices = tf.where(objectiness == 0).numpy()    # without this == > ablation 2
        only_background_logits = tf.gather(only_background_output, only_background_indices) # without this == > ablation 2
        only_background_loss = tf.reduce_mean( -tf.math.log(only_background_logits + tf.keras.backend.epsilon()) )  # without this == > ablation 2
        ##############################################################################################################################

        total_loss = object_loss + crop_weed_dice_loss + crop_weed_distri_loss + only_background_loss + only_one_loss
        # total_loss = object_loss + crop_weed_dice_loss + crop_weed_distri_loss

    return total_loss, logits_


# yilog(h(xi;θ))+(1−yi)log(1−h(xi;θ))
def main():
    tf.keras.backend.clear_session()

    model = Unet(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), classes=1)
    model2 = fix_Unet(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), classes=1)    # model is too heavy

    # Tomorrow need to fix layers!!! remember!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
    model.summary()
    model2.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, model2=model2, optim=optim, optim2=optim2)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!")
    
    if FLAGS.train:
        count = 0
        output_text = open(FLAGS.save_print, "w")
        
        train_list = np.loadtxt(FLAGS.train_txt_path, dtype="<U200", skiprows=0, usecols=0)
        val_list = np.loadtxt(FLAGS.val_txt_path, dtype="<U200", skiprows=0, usecols=0)
        test_list = np.loadtxt(FLAGS.test_txt_path, dtype="<U200", skiprows=0, usecols=0)

        train_img_dataset = [FLAGS.image_path + data for data in train_list]
        val_img_dataset = [FLAGS.image_path + data for data in val_list]
        test_img_dataset = [FLAGS.image_path + data for data in test_list]

        train_lab_dataset = [FLAGS.label_path + data for data in train_list]
        val_lab_dataset = [FLAGS.label_path + data for data in val_list]
        test_lab_dataset = [FLAGS.label_path + data for data in test_list]

        val_ge = tf.data.Dataset.from_tensor_slices((val_img_dataset, val_lab_dataset))
        val_ge = val_ge.map(test_func)
        val_ge = val_ge.batch(1)
        val_ge = val_ge.prefetch(tf.data.experimental.AUTOTUNE)

        test_ge = tf.data.Dataset.from_tensor_slices((test_img_dataset, test_lab_dataset))
        test_ge = test_ge.map(test_func)
        test_ge = test_ge.batch(1)
        test_ge = test_ge.prefetch(tf.data.experimental.AUTOTUNE)

        tr_loss = open(FLAGS.train_loss_graphs, "w")
        tr_acc = open(FLAGS.train_acc_graphs, "w")
        val_loss = open(FLAGS.val_loss_graphs, "w")
        val_acc = open(FLAGS.val_acc_graphs, "w")

        for epoch in range(FLAGS.epochs):
            A = list(zip(train_img_dataset, train_lab_dataset))
            shuffle(A)
            train_img_dataset, train_lab_dataset = zip(*A)
            train_img_dataset, train_lab_dataset = np.array(train_img_dataset), np.array(train_lab_dataset)

            train_ge = tf.data.Dataset.from_tensor_slices((train_img_dataset, train_lab_dataset))
            train_ge = train_ge.shuffle(len(train_img_dataset))
            train_ge = train_ge.map(tr_func)
            train_ge = train_ge.batch(FLAGS.batch_size)
            train_ge = train_ge.prefetch(tf.data.experimental.AUTOTUNE)
            tr_iter = iter(train_ge)

            tr_iter = iter(train_ge)
            tr_idx = len(train_img_dataset) // FLAGS.batch_size
            train_loss = 0.
            for step in range(tr_idx):
                batch_images, print_images, batch_labels = next(tr_iter)

                batch_labels = batch_labels.numpy()
                batch_labels = np.where(batch_labels == FLAGS.ignore_label, 2, batch_labels)    # 2 is void
                batch_labels = np.where(batch_labels == 255, 0, batch_labels)
                batch_labels = np.where(batch_labels == 128, 1, batch_labels)
                batch_labels = np.squeeze(batch_labels, -1)

                class_imbal_labels_buf = 0.
                class_imbal_labels = batch_labels
                for i in range(FLAGS.batch_size):
                    class_imbal_label = class_imbal_labels[i]
                    class_imbal_label = np.reshape(class_imbal_label, [FLAGS.img_size*FLAGS.img_size, ])
                    count_c_i_lab = np.bincount(class_imbal_label, minlength=FLAGS.total_classes)
                    class_imbal_labels_buf += count_c_i_lab

                class_imbal_labels_buf = class_imbal_labels_buf[0:FLAGS.total_classes]
                object_buf = np.array([class_imbal_labels_buf[2], class_imbal_labels_buf[0] + class_imbal_labels_buf[1]], dtype=np.float32)
                crop_buf = np.array([class_imbal_labels_buf[1], class_imbal_labels_buf[0]], dtype=np.float32)
                weed_buf = np.array([class_imbal_labels_buf[0], class_imbal_labels_buf[1]], dtype=np.float32)

                class_imbal_labels_buf = (np.max(class_imbal_labels_buf / np.sum(class_imbal_labels_buf)) + 1 - (class_imbal_labels_buf / np.sum(class_imbal_labels_buf)))
                object_buf = (np.max(object_buf / np.sum(object_buf)) + 1 - (object_buf / np.sum(object_buf)))
                crop_buf = (np.max(crop_buf / np.sum(crop_buf)) + 1 - (crop_buf / np.sum(crop_buf)))
                weed_buf = (np.max(weed_buf / np.sum(weed_buf)) + 1 - (weed_buf / np.sum(weed_buf)))

                class_imbal_labels_buf = tf.nn.softmax(class_imbal_labels_buf[0:FLAGS.total_classes-1]).numpy()
                object_buf = tf.nn.softmax(object_buf).numpy()
                crop_buf = tf.nn.softmax(crop_buf).numpy()
                weed_buf = tf.nn.softmax(weed_buf).numpy()

                objectiness = np.where(batch_labels == 2, 0, 1)  # 피사체가 있는곳은 1 없는곳은 0으로 만들어준것

                loss = cal_loss(model, model2, batch_images, batch_labels, objectiness, class_imbal_labels_buf, object_buf, crop_buf, weed_buf)
                train_loss += loss

                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] loss = {}".format(epoch, step+1, tr_idx, loss))

                if count % 100 == 0:

                    raw_logits = run_model(model, batch_images, False)
                    output = run_model(model2, batch_images * tf.nn.sigmoid(raw_logits), False)
                    object_output = tf.nn.sigmoid(output[:, :, :, 1])
                    crop_weed_output = tf.nn.sigmoid(output[:, :, :, 0])
                    for i in range(FLAGS.batch_size):
                        label = tf.cast(batch_labels[i], tf.int32).numpy()
                        object_image = object_output[i]
                        object_image = tf.where(object_image >= 0.5, 1, 0).numpy()
                        false_object_indices = np.where(object_image == 0)

                        crop_weed_image = tf.where(crop_weed_output[i] >= 0.5, 1, 0).numpy()
                        crop_weed_image[false_object_indices] = 2
                        image = crop_weed_image

                        pred_mask_color = color_map[crop_weed_image]
                        label_mask_color = color_map[label]

                        temp_img = np.concatenate((image[:, :, np.newaxis], image[:, :, np.newaxis], image[:, :, np.newaxis]), -1)
                        image = np.concatenate((image[:, :, np.newaxis], image[:, :, np.newaxis], image[:, :, np.newaxis]), -1)
                        pred_mask_warping = np.where(temp_img == np.array([2,2,2], dtype=np.uint8), print_images[i], image)
                        pred_mask_warping = np.where(temp_img == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), pred_mask_warping)
                        pred_mask_warping = np.where(temp_img == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), pred_mask_warping)
                        pred_mask_warping /= 255.
 
                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_label.png", label_mask_color)
                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_predict.png", pred_mask_color)
                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_warping_predict.png", pred_mask_warping)
                    

                count += 1
            train_loss /= tr_idx
            tr_loss.write(str(train_loss.numpy()))
            tr_loss.write("\n")
            tr_loss.flush()

            tr_iter = iter(train_ge)
            miou = 0.
            f1_score_ = 0.
            crop_iou = 0.
            weed_iou = 0.
            recall_ = 0.
            precision_ = 0.
            for i in range(tr_idx):
                batch_images, _, batch_labels = next(tr_iter)
                for j in range(FLAGS.batch_size):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    raw_logits = run_model(model, batch_image, False)
                    output = run_model(model2, batch_image * tf.nn.sigmoid(raw_logits), False)
                    object_output = tf.nn.sigmoid(output[0, :, :, 1])
                    object_output = tf.where(object_output >= 0.5, 1, 0).numpy()
                    false_object_indices = np.where(object_output == 0)

                    crop_weed_output = tf.where(output[0, :, :, 0] >= 0.5, 1, 0).numpy()
                    crop_weed_output[false_object_indices] = 2
                    image = crop_weed_output

                    batch_label = batch_labels[j]
                    batch_label = tf.cast(batch_labels[j], tf.uint8).numpy()
                    batch_label = np.where(batch_label == FLAGS.ignore_label, 2, batch_label)    # 2 is void
                    batch_label = np.where(batch_label == 255, 0, batch_label)
                    batch_label = np.where(batch_label == 128, 1, batch_label)

                    miou_, crop_iou_, weed_iou_ = Measurement(predict=image,
                                        label=batch_label, 
                                        shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                        total_classes=FLAGS.total_classes).MIOU()
                    
                    miou += miou_
                    crop_iou += crop_iou_
                    weed_iou += weed_iou_

            miou_ = miou[0,0]/(miou[0,0] + miou[0,1] + miou[1,0])
            crop_iou_ = crop_iou[0,0]/(crop_iou[0,0] + crop_iou[0,1] + crop_iou[1,0])
            weed_iou_ = weed_iou[0,0]/(weed_iou[0,0] + weed_iou[0,1] + weed_iou[1,0])
            recall_ = miou[0,0] / (miou[0,0] + miou[0,1])
            precision_ = miou[0,0] / (miou[0,0] + miou[1,0])
            f1_score_ = (2*precision_*recall_) / (precision_ + recall_)
            accurcay = (miou[0,0] + miou[1,1]) / (miou[0,0] + miou[0,1] + miou[1,0] + miou[1,1])
            accurcay = accurcay * 100.
            print("train mIoU = %.4f (crop_iou = %.4f, weed_iou = %.4f), train F1_score = %.4f, train sensitivity(recall) = %.4f, train precision = %.4f" % (miou_,
                                                                                                                                                crop_iou_,
                                                                                                                                                weed_iou_,
                                                                                                                                                f1_score_,
                                                                                                                                                recall_,
                                                                                                                                                precision_))
            output_text.write("Epoch: ")
            output_text.write(str(epoch))
            output_text.write("===================================================================")
            output_text.write("\n")
            output_text.write("train mIoU: ")
            output_text.write("%.4f" % (miou_))
            output_text.write(", crop_iou: ")
            output_text.write("%.4f" % (crop_iou_))
            output_text.write(", weed_iou: ")
            output_text.write("%.4f" % (weed_iou_))
            output_text.write(", train F1_score: ")
            output_text.write("%.4f" % (f1_score_))
            output_text.write(", train sensitivity: ")
            output_text.write("%.4f" % (recall_))
            output_text.write("\n")

            tr_acc.write(str(accurcay))
            tr_acc.write("\n")
            tr_acc.flush()

            val_iter = iter(val_ge)
            miou = 0.
            f1_score_ = 0.
            crop_iou = 0.
            weed_iou = 0.
            recall_ = 0.
            precision_ = 0.
            total_v_loss = 0.
            # val_model = Unet(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), classes=1)
            # val_model2 = fix_Unet(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), classes=1)    # model is too heavy

            # val_model.set_weights(model.get_weights())
            # val_model2.set_weights(model2.get_weights())
            for i in range(len(val_img_dataset)):
                batch_images, batch_labels = next(val_iter)

                temp_batch_labels = batch_labels
                temp_batch_labels = temp_batch_labels.numpy()
                temp_batch_labels = np.where(temp_batch_labels == FLAGS.ignore_label, 2, temp_batch_labels)    # 2 is void
                temp_batch_labels = np.where(temp_batch_labels == 255, 0, temp_batch_labels)
                temp_batch_labels = np.where(temp_batch_labels == 128, 1, temp_batch_labels)
                temp_batch_labels = np.squeeze(temp_batch_labels, -1)

                class_imbal_labels_buf = 0.
                class_imbal_labels = temp_batch_labels
                for i in range(1):
                    class_imbal_label = class_imbal_labels[i]
                    class_imbal_label = np.reshape(class_imbal_label, [FLAGS.img_size*FLAGS.img_size, ])
                    count_c_i_lab = np.bincount(class_imbal_label, minlength=FLAGS.total_classes)
                    class_imbal_labels_buf += count_c_i_lab

                class_imbal_labels_buf = class_imbal_labels_buf[0:FLAGS.total_classes]
                object_buf = np.array([class_imbal_labels_buf[2], class_imbal_labels_buf[0] + class_imbal_labels_buf[1]], dtype=np.float32)
                crop_buf = np.array([class_imbal_labels_buf[1], class_imbal_labels_buf[0]], dtype=np.float32)
                weed_buf = np.array([class_imbal_labels_buf[0], class_imbal_labels_buf[1]], dtype=np.float32)

                class_imbal_labels_buf = (np.max(class_imbal_labels_buf / np.sum(class_imbal_labels_buf)) + 1 - (class_imbal_labels_buf / np.sum(class_imbal_labels_buf)))
                object_buf = (np.max(object_buf / np.sum(object_buf)) + 1 - (object_buf / np.sum(object_buf)))
                crop_buf = (np.max(crop_buf / np.sum(crop_buf)) + 1 - (crop_buf / np.sum(crop_buf)))
                weed_buf = (np.max(weed_buf / np.sum(weed_buf)) + 1 - (weed_buf / np.sum(weed_buf)))

                class_imbal_labels_buf = tf.nn.softmax(class_imbal_labels_buf[0:FLAGS.total_classes-1]).numpy()
                object_buf = tf.nn.softmax(object_buf).numpy()
                crop_buf = tf.nn.softmax(crop_buf).numpy()
                weed_buf = tf.nn.softmax(weed_buf).numpy()

                objectiness = np.where(temp_batch_labels == 2, 0, 1)  # 피사체가 있는곳은 1 없는곳은 0으로 만들어준것

                v_loss, output = val_cal_loss(model, model2, batch_images, temp_batch_labels, objectiness, class_imbal_labels_buf, object_buf, crop_buf, weed_buf)
                total_v_loss += v_loss

                for j in range(1):
                    # batch_image = tf.expand_dims(batch_images[j], 0)
                    # raw_logits = run_model(model, batch_image, False)
                    # output = run_model(model2, batch_image * tf.nn.sigmoid(raw_logits), False)
                    object_output = tf.nn.sigmoid(output[0, :, :, 1])
                    object_output = tf.where(object_output >= 0.5, 1, 0).numpy()
                    false_object_indices = np.where(object_output == 0)

                    crop_weed_output = tf.where(output[0, :, :, 0] >= 0.5, 1, 0).numpy()
                    crop_weed_output[false_object_indices] = 2
                    image = crop_weed_output

                    batch_label = batch_labels[j]
                    batch_label = tf.cast(batch_labels[j], tf.uint8).numpy()
                    batch_label = np.where(batch_label == FLAGS.ignore_label, 2, batch_label)    # 2 is void
                    batch_label = np.where(batch_label == 255, 0, batch_label)
                    batch_label = np.where(batch_label == 128, 1, batch_label)

                    miou_, crop_iou_, weed_iou_ = Measurement(predict=image,
                                        label=batch_label, 
                                        shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                        total_classes=FLAGS.total_classes).MIOU()
                    
                    miou += miou_
                    crop_iou += crop_iou_
                    weed_iou += weed_iou_

            total_v_loss /= len(val_img_dataset)
            val_loss.write(str(total_v_loss.numpy()))
            val_loss.write("\n")
            val_loss.flush()
            
            miou_ = miou[0,0]/(miou[0,0] + miou[0,1] + miou[1,0])
            crop_iou_ = crop_iou[0,0]/(crop_iou[0,0] + crop_iou[0,1] + crop_iou[1,0])
            weed_iou_ = weed_iou[0,0]/(weed_iou[0,0] + weed_iou[0,1] + weed_iou[1,0])
            recall_ = miou[0,0] / (miou[0,0] + miou[0,1])
            precision_ = miou[0,0] / (miou[0,0] + miou[1,0])
            f1_score_ = (2*precision_*recall_) / (precision_ + recall_)
            accurcay = (miou[0,0] + miou[1,1]) / (miou[0,0] + miou[0,1] + miou[1,0] + miou[1,1])
            accurcay = accurcay * 100.
            print("val mIoU = %.4f (crop_iou = %.4f, weed_iou = %.4f), val F1_score = %.4f, val sensitivity(recall) = %.4f, val precision = %.4f" % (miou_,
                                                                                                                                                crop_iou_,
                                                                                                                                                weed_iou_,
                                                                                                                                                f1_score_,
                                                                                                                                                recall_,
                                                                                                                                                precision_))
            output_text.write("val mIoU: ")
            output_text.write("%.4f" % (miou_))
            output_text.write(", crop_iou: ")
            output_text.write("%.4f" % (crop_iou_))
            output_text.write(", weed_iou: ")
            output_text.write("%.4f" % (weed_iou_))
            output_text.write(", val F1_score: ")
            output_text.write("%.4f" % (f1_score_))
            output_text.write(", val sensitivity: ")
            output_text.write("%.4f" % (recall_))
            output_text.write("\n")

            val_acc.write(str(accurcay))
            val_acc.write("\n")
            val_acc.flush()

            test_iter = iter(test_ge)
            miou = 0.
            f1_score_ = 0.
            crop_iou = 0.
            weed_iou = 0.
            recall_ = 0.
            precision_ = 0.
            for i in range(len(test_img_dataset)):
                batch_images, batch_labels = next(test_iter)
                for j in range(1):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    raw_logits = run_model(model, batch_image, False)
                    output = run_model(model2, batch_image * tf.nn.sigmoid(raw_logits), False)
                    object_output = tf.nn.sigmoid(output[0, :, :, 1])
                    object_output = tf.where(object_output >= 0.5, 1, 0).numpy()
                    false_object_indices = np.where(object_output == 0)

                    crop_weed_output = tf.where(output[0, :, :, 0] >= 0.5, 1, 0).numpy()
                    crop_weed_output[false_object_indices] = 2
                    image = crop_weed_output

                    batch_label = batch_labels[j]
                    batch_label = tf.cast(batch_labels[j], tf.uint8).numpy()
                    batch_label = np.where(batch_label == FLAGS.ignore_label, 2, batch_label)    # 2 is void
                    batch_label = np.where(batch_label == 255, 0, batch_label)
                    batch_label = np.where(batch_label == 128, 1, batch_label)

                    miou_, crop_iou_, weed_iou_ = Measurement(predict=image,
                                        label=batch_label, 
                                        shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                        total_classes=FLAGS.total_classes).MIOU()

                    miou += miou_
                    crop_iou += crop_iou_
                    weed_iou += weed_iou_

            miou_ = miou[0,0]/(miou[0,0] + miou[0,1] + miou[1,0])
            crop_iou_ = crop_iou[0,0]/(crop_iou[0,0] + crop_iou[0,1] + crop_iou[1,0])
            weed_iou_ = weed_iou[0,0]/(weed_iou[0,0] + weed_iou[0,1] + weed_iou[1,0])
            recall_ = miou[0,0] / (miou[0,0] + miou[0,1])
            precision_ = miou[0,0] / (miou[0,0] + miou[1,0])
            f1_score_ = (2*precision_*recall_) / (precision_ + recall_)
            print("test mIoU = %.4f (crop_iou = %.4f, weed_iou = %.4f), test F1_score = %.4f, test sensitivity(recall) = %.4f, test precision = %.4f" % (miou_,
                                                                                                                                                crop_iou_,
                                                                                                                                                weed_iou_,
                                                                                                                                                f1_score_,
                                                                                                                                                recall_,
                                                                                                                                                precision_))
            print("=================================================================================================================================================")
            output_text.write("test mIoU: ")
            output_text.write("%.4f" % (miou_))
            output_text.write(", crop_iou: ")
            output_text.write("%.4f" % (crop_iou_))
            output_text.write(", weed_iou: ")
            output_text.write("%.4f" % (weed_iou_))
            output_text.write(", test F1_score: ")
            output_text.write("%.4f" % (f1_score_))
            output_text.write(", test sensitivity: ")
            output_text.write("%.4f" % (recall_))
            output_text.write("\n")
            output_text.write("===================================================================")
            output_text.write("\n")
            output_text.flush()

            model_dir = "%s/%s" % (FLAGS.save_checkpoint, epoch)
            if not os.path.isdir(model_dir):
                print("Make {} folder to store the weight!".format(epoch))
                os.makedirs(model_dir)
            ckpt = tf.train.Checkpoint(model=model, model2=model2, optim=optim, optim2=optim2)
            ckpt_dir = model_dir + "/Crop_weed_model_{}.ckpt".format(epoch)
            ckpt.save(ckpt_dir)
    else:
        test_list = np.loadtxt(FLAGS.test_txt_path, dtype="<U200", skiprows=0, usecols=0)

        test_img_dataset = [FLAGS.image_path + data for data in test_list]
        test_lab_dataset = [FLAGS.label_path + data for data in test_list]

        test_ge = tf.data.Dataset.from_tensor_slices((test_img_dataset, test_lab_dataset))
        test_ge = test_ge.map(test_func2)
        test_ge = test_ge.batch(1)
        test_ge = test_ge.prefetch(tf.data.experimental.AUTOTUNE)

        test_iter = iter(test_ge)
        miou = 0.
        f1_score_ = 0.
        crop_iou = 0.
        weed_iou = 0.
        recall_ = 0.
        precision_ = 0.
        for i in range(len(test_img_dataset)):
            batch_images, nomral_img, batch_labels = next(test_iter)
            batch_labels = tf.squeeze(batch_labels, -1)
            for j in range(1):
                batch_image = tf.expand_dims(batch_images[j], 0)
                raw_logits = run_model(model, batch_image, False)
                output = run_model(model2, batch_image * tf.nn.sigmoid(raw_logits), False)
                object_output = tf.nn.sigmoid(output[0, :, :, 1])
                object_output = tf.where(object_output >= 0.5, 1, 0).numpy()
                false_object_indices = np.where(object_output == 0)

                crop_weed_output = tf.where(output[0, :, :, 0] >= 0.5, 1, 0).numpy()
                crop_weed_output[false_object_indices] = 2
                image = crop_weed_output

                batch_label = batch_labels[j]
                batch_label = tf.cast(batch_labels[j], tf.uint8).numpy()
                batch_label = np.where(batch_label == FLAGS.ignore_label, 2, batch_label)    # 2 is void
                batch_label = np.where(batch_label == 255, 0, batch_label)
                batch_label = np.where(batch_label == 128, 1, batch_label)

                miou_, crop_iou_, weed_iou_ = Measurement(predict=image,
                                    label=batch_label, 
                                    shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                    total_classes=FLAGS.total_classes).MIOU()

                pred_mask_color = color_map[crop_weed_output]  # 논문그림처럼 할것!
                batch_label = np.expand_dims(batch_label, -1)
                batch_label = np.concatenate((batch_label, batch_label, batch_label), -1)
                label_mask_color = np.zeros([FLAGS.img_size, FLAGS.img_size, 3], dtype=np.uint8)
                label_mask_color = np.where(batch_label == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), label_mask_color)
                label_mask_color = np.where(batch_label == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), label_mask_color)

                temp_img = np.concatenate((crop_weed_output[:, :, np.newaxis], crop_weed_output[:, :, np.newaxis], crop_weed_output[:, :, np.newaxis]), -1)
                image = np.concatenate((image[:, :, np.newaxis], image[:, :, np.newaxis], image[:, :, np.newaxis]), -1)
                pred_mask_warping = np.where(temp_img == np.array([2,2,2], dtype=np.uint8), nomral_img[j], image)
                pred_mask_warping = np.where(temp_img == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), pred_mask_warping)
                pred_mask_warping = np.where(temp_img == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), pred_mask_warping)
                pred_mask_warping /= 255.

                name = test_img_dataset[i].split("/")[-1].split(".")[0]
                plt.imsave(FLAGS.test_images + "/" + name + "_label.png", label_mask_color)
                plt.imsave(FLAGS.test_images + "/" + name + "_predict.png", pred_mask_color)
                plt.imsave(FLAGS.test_images + "/" + name + "_predict_warp.png", pred_mask_warping)
                #back_iou += back_iou_

                miou += miou_
                crop_iou += crop_iou_
                weed_iou += weed_iou_

        miou_ = miou[0,0]/(miou[0,0] + miou[0,1] + miou[1,0])
        crop_iou_ = crop_iou[0,0]/(crop_iou[0,0] + crop_iou[0,1] + crop_iou[1,0])
        weed_iou_ = weed_iou[0,0]/(weed_iou[0,0] + weed_iou[0,1] + weed_iou[1,0])
        recall_ = miou[0,0] / (miou[0,0] + miou[0,1])
        precision_ = miou[0,0] / (miou[0,0] + miou[1,0])
        f1_score_ = (2*precision_*recall_) / (precision_ + recall_)
        print("test mIoU = %.4f (crop_iou = %.4f, weed_iou = %.4f), test F1_score = %.4f, test sensitivity(recall) = %.4f, test precision = %.4f" % (miou_,
                                                                                                                                            crop_iou_,
                                                                                                                                            weed_iou_,
                                                                                                                                            f1_score_,
                                                                                                                                            recall_,
                                                                                                                                            precision_))

if __name__ == "__main__":
    main()
