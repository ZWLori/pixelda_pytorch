import params
import torch
import model


def add_task_specific_model(images_size, num_classes):
    """
    Create a classifier for the given images
    :return: the logits, a tensor of shape [batch_size, num_classes]
    """
    task = params.task_tower
    if task == 'doubling_pose_estimator':
        # todo get logits, quaternion_pred from it
        doubling_pose_estimator = model.DoublingCNNAndQuaternion(images_size, params.num_private_layers, num_classes)
    else:
        raise ValueError('Undefined task classifier %s' % task)
    return doubling_pose_estimator


def create_model(
        target_images_shape=None,
        source_images_shape=None,
        noise=None,
        num_classes=None):
    """Create a GAN model.

      Arguments:
        hparams: HParam object specifying model params
        target_images: A `Tensor` of size [batch_size, height, width, channels]. It
          is assumed that the images are [-1, 1] normalized.
        source_images: A `Tensor` of size [batch_size, height, width, channels]. It
          is assumed that the images are [-1, 1] normalized.
        source_labels: A `Tensor` of size [batch_size] of categorical labels between
          [0, num_classes]
        is_training: whether model is currently training
        noise: If None, model generates its own noise. Otherwise use provided.
        num_classes: Number of classes for classification

      Returns:
        components dictionary

      Raises:
        ValueError: unknown hparams.arch setting
      """
    if num_classes is None:
        raise ValueError('Num classes must be provided to create task classifier')

    if params.arch not in ['resnet', 'simple', 'residual_interpretation']:
        raise ValueError('Undefined architecture %s' %params.arch)

    components = {}

    # todo enable the following session later
    ###########################
    # Create latent variables #
    ###########################
    latent_vars = dict()
    # disable noise channel first
    if params.noise_channel:
        noise_shape = [params.batch_size, params.noise_dims]
        if noise is not None:
            assert list(noise.size()) == noise_shape
            print('Using provided noise')
        else:
            print('Using random noise')
            noise = torch.FloatTensor(noise_shape).uniform_(-1, 1)
        latent_vars['noise'] = noise

    ####################
    # Create generator #
    ####################
    if params.arch == 'resnet':
        generator = model.ResnetGenerator(source_images_shape, list(target_images_shape)[1:4], latent_vars=latent_vars)
    elif params.arch == 'residual_interpretation':
        generator = model.ResidualInterpretationGenerator(source_images_shape, latent_vars=latent_vars)
    elif params.arch == 'simple':
        generator = model.SimpleGenerator(source_images_shape, target_images_shape, latent_vars)
    else:
        raise ValueError('Undefined architecture')

    components['generator'] = generator

    #####################
    # Domain Classifier #
    #####################
    # todo assume all the images have same shape
    transferred_images_shape = target_images_shape
    components['transferred_domain_logits'] = model.Discriminator(transferred_images_shape)
    components['target_domain_logits'] = model.Discriminator(target_images_shape)

    ###################
    # Task Classifier #
    ###################
    components['source_task_classifier'] = add_task_specific_model(source_images_shape, num_classes)
    components['target_task_classifier'] = add_task_specific_model(target_images_shape, num_classes)
    components['transferred_task_classifier'] = add_task_specific_model(transferred_images_shape, num_classes)

    return components

