import torch
import torch.nn.modules.loss as loss
import params


def get_domain_classifier_losses(transfer_pred, transfer_label, target_pred, target_label):
    """
    Losses replated to the domain-classifier
    :return: loss, a tensor representing the total task-classfier loss
    """
    if params.domain_loss_weight == 0:
        print("Domain classifier loss weight is 0.")
        return 0

    transferred_criterion = loss.MultiLabelSoftMarginLoss()
    transferred_domain_loss = transferred_criterion(transfer_pred, transfer_label)

    target_criterion = loss.MultiLabelSoftMarginLoss()
    target_domain_loss = target_criterion(target_pred, target_label)

    total_domain_loss = transferred_domain_loss + target_domain_loss
    total_domain_loss *= params.domain_loss_weight

    print("Domain loss = %s" % total_domain_loss)
    return total_domain_loss



def get_task_specific_losses(source_labels, num_classes, source_task=None, transfer_task=None):
    """
    Losses related to the task-classifier
    """
    print(source_labels.size())
    one_hot_labels = torch.zero(list(source_labels.size())[0], num_classes).scatter_(1, source_labels, 1)

    task_specific_loss = 0
    if source_task is not None:
        source_criterion = loss.CrossEntropyLoss()
        source_loss = source_criterion(one_hot_labels, source_task, weight=params.source_task_loss_weight)
        task_specific_loss += source_loss
    if transfer_task is not None:
        transfer_criterion = loss.CrossEntropyLoss()
        transfer_loss = transfer_criterion(one_hot_labels, transfer_task, weight=params.transferred_task_loss_weight)
        task_specific_loss += transfer_loss

    print("Task specific loss = %s" %task_specific_loss)

    return task_specific_loss



def transfer_similarity_loss(reconstructions, source_images, weight):
    """
    Computes a loss encouraging similarity between source and transferred.
    """
    #todo ???? shouldnt be similarity btw target and transfer???

    if weight == 0:
        return

    reconstruction_similarity_criterion = loss.MSELoss()
    # todo : check pairewise mse
    reconstruction_similarity_loss = reconstruction_similarity_criterion(reconstructions, source_images, weight)

    return reconstruction_similarity_loss

def g_step_loss(source_images, source_labels, transferred_images, transferred_domain_logits, num_classes):
    """
    Configure the loss function which runs during the generation step
    :return:
    """

    generator_loss = 0
    style_transfer_criterion = loss.CrossEntropyLoss()
    style_transfer_loss = style_transfer_criterion(transferred_domain_logits, source_labels, weight=params.style_transfer_loss_weight)

    generator_loss += style_transfer_loss

    ###########################
    # Content Similarity Loss #
    ###########################
    generator_loss += transfer_similarity_loss(transferred_images,
                                               source_images,
                                               params.transferred_similarity_loss_weight)

    # optimize the style transfer network to maximize classificaton accuracy
    if source_labels is not None and params.task_tower_in_g_step:
        # todo specify source_task as well as transfer_task
        generator_loss += get_task_specific_losses(source_labels, num_classes) * params.task_loss_in_g_weight

    return generator_loss

def d_step_loss(transfer_pred, transfer_label, target_pred, target_label, source_labels, num_classes):
    """
    Configure the loss function which runs during the discrimination step
    Note that during the d-step, the model optimizes both the domain classifier and the task classifier
    :return:
    """
    ######################
    #    Domain Loss     #
    ######################
    domain_classifier_loss = get_domain_classifier_losses(transfer_pred, transfer_label, target_pred, target_label)

    ######################
    # Task Specific Loss #
    ######################
    task_specific_loss = 0
    if source_labels is not None:
        task_specific_loss = get_task_specific_losses(source_labels, num_classes)

    return domain_classifier_loss + task_specific_loss




