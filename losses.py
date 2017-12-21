import torch
import torch.nn.modules.loss as loss
import params
import torch.nn as nn
from torch.autograd import Variable


def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    target = Variable(target)

    return target


def get_domain_classifier_losses(trans_domain_logits, target_domain_logits):
    """
    Losses replated to the domain-classifier
    :return: loss, a tensor representing the total task-classfier loss
    """
    if params.domain_loss_weight == 0:
        print("Domain classifier loss weight is 0.")
        return 0
    transfer_label = torch.ones_like(trans_domain_logits)
    transferred_criterion = loss.MultiLabelSoftMarginLoss()
    transferred_domain_loss = transferred_criterion(trans_domain_logits, transfer_label)

    target_label = torch.ones_like(target_domain_logits)
    target_criterion = loss.MultiLabelSoftMarginLoss()
    target_domain_loss = target_criterion(target_domain_logits, target_label)

    total_domain_loss = transferred_domain_loss + target_domain_loss
    total_domain_loss *= params.domain_loss_weight

    return total_domain_loss


def get_task_specific_losses(source_labels, source_task=None, transfer_task=None):
    """
    Losses related to the task-classifier
    """
    # source_labels = source_labels.view(params.batch_size, 1)
    # one_hot_labels = torch.zeros(params.batch_size, params.num_classes).scatter_(1, source_labels.data, 1)

    task_specific_loss = 0
    if source_task is not None:
        source_criterion = loss.CrossEntropyLoss()
        source_loss = source_criterion(source_task, source_labels)
        task_specific_loss += source_loss
    if transfer_task is not None:
        transfer_criterion = loss.CrossEntropyLoss()
        transfer_loss = transfer_criterion(transfer_task, source_labels)
        task_specific_loss += transfer_loss

    return task_specific_loss


def transfer_similarity_loss(reconstructions, source_images, weight):
    """
    Computes a loss encouraging similarity between source and transferred.
    """
    #todo ???? shouldnt be similarity btw target and transfer???

    if weight == 0:
        return 0

    reconstruction_similarity_criterion = loss.MSELoss()
    # todo : check pairewise mse
    reconstruction_similarity_loss = reconstruction_similarity_criterion(reconstructions, source_images)

    return reconstruction_similarity_loss


def g_step_loss(source_images, source_labels, source_task_logits, trans_images, trans_domain_logits, trans_task_logits):
    """
    Configure the loss function which runs during the generation step
    """
    generator_loss = 0
    style_transfer_criterion = nn.MultiLabelSoftMarginLoss()
    style_transfer_loss = style_transfer_criterion(trans_domain_logits, torch.ones_like(trans_domain_logits))

    generator_loss += style_transfer_loss

    ###########################
    # Content Similarity Loss #
    ###########################
    generator_loss += transfer_similarity_loss(trans_images,
                                               source_images,
                                               params.transferred_similarity_loss_weight)

    # optimize the style transfer network to maximize classification accuracy
    ######################
    # Task Specific Loss #
    ######################
    if source_labels is not None and params.task_tower_in_g_step:
        task_specific_losses = get_task_specific_losses(source_labels, source_task_logits, trans_task_logits) \
                               * params.task_loss_in_g_weight
        generator_loss += task_specific_losses

    print("g_step_loss %f" % generator_loss.data)
    return generator_loss


def d_step_loss(trans_task_logits, trans_domain_logits, target_domain_logits, source_labels, source_task_logits):
    """
    Configure the loss function which runs during the discrimination step
    Note that during the d-step, the model optimizes both the domain classifier and the task classifier
    """
    discriminator_loss = 0
    ######################
    #    Domain Loss     #
    ######################
    discriminator_loss += get_domain_classifier_losses(trans_domain_logits, target_domain_logits)

    ######################
    # Task Specific Loss #
    #     transfer       #
    ######################
    if source_labels is not None:
        discriminator_loss += get_task_specific_losses(source_labels, source_task_logits, trans_task_logits)

    print("d_step_loss %f" % discriminator_loss.data)
    return discriminator_loss




