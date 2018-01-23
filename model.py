import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import params


def flatten(tensor):
    return tensor.view(tensor.data.size(0), -1)


def calculate_flat_fts(in_size, features):
    f = features(Variable(torch.ones(1, *in_size[1:])))
    return int(np.prod(f.size()[1:]))


def project_latent_vars(proj_shape, latent_vars, combine_method='sum'):
    """
    Generate noise and project to input volume size.

    :param:
        proj_shape: Shape to project noise (not including batch size).
        latent_vars: dictionary of `'key': Tensor of shape [batch_size, N]`
        combine_method: How to combine the projected values.
          sum = project to volume then sum
          concat = concatenate along last dimension (i.e. channel)

    :return:
        If combine_method=sum, a `Tensor` of size `hparams.projection_shape`
        If combine_method=concat and there are N latent vars, a `Tensor` of size
          `hparams.projection_shape`, with the last channel multiplied by N

    :raise:
        ValueError: combine_method is not one of sum/concat
      """
    values = []
    for var in latent_vars:
        # Project & reshape noise to a HxWxC input
        projected = nn.Sequential(nn.Linear(latent_vars[var], np.prod(proj_shape)),
                                  nn.BatchNorm2d(np.prod(proj_shape)),
                                  nn.ReLU)

        values.append(projected.view(params.batch_size + proj_shape))
    if combine_method == 'sum':
        result = sum(values)
    elif combine_method == 'concat':
        result = torch.cat(values, len(proj_shape))
    else:
        raise ValueError("Unkown combine_method &s" %combine_method)

    return result


class SimpleGenerator(nn.Module):
    """Simple generator architecture (stack of convs) for trying small models."""
    def __init__(self, source_image_shape, target_image_shape, latent_vars):
        super(SimpleGenerator, self).__init__()
        self.target_size = target_image_shape
        self.latent_vars = latent_vars

        ###################################################
        # Transfer the source images to the target style. #
        ###################################################
        self.conv_layers = []
        in_channel = list(source_image_shape)[1]
        for i in range(0, params.simple_num_conv_layers):
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channel, params.simple_conv_filters, kernal_size=params.generator_kernel_size),
                nn.BatchNorm2d(params.simple_conv_filters),
                nn.RelU()
            )
            in_channel = params.simple_conv_filters

        # Project back to the right # image channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, list(target_image_shape)[1], kernal_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        # disabled latent vars first
        if self.latent_vars:
            projected_latent = project_latent_vars(proj_shape=list(x.size())[1:3] + [1],
                                                   latent_vars=self.latent_vars,
                                                   combine_method='concat')
            feed_x = torch.cat([x, projected_latent], 1)
        conv_layers_out = x
        for layer in self.conv_layers:
            conv_layers_out = layer(conv_layers_out)

        transferred_imgs = self.conv2(conv_layers_out)
        assert transferred_imgs.size() == self.target.size()

        return transferred_imgs


class ResidualBlock(nn.Module):
    """
    Create a resnet block
    """
    def __init__(self, in_dim):
        print("ResidualBlock")
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, params.resnet_filters, kernel_size=params.generator_kernel_size, padding=(params.generator_kernel_size-1)//2),
            nn.BatchNorm2d(params.resnet_filters),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(params.resnet_filters, params.resnet_filters, kernel_size=params.generator_kernel_size, padding=(params.generator_kernel_size-1)//2),
            nn.BatchNorm2d(params.resnet_filters),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if params.resnet_residuals:
            out += x
        return out


class ResnetStack(nn.Module):
    """
    Create a resnet style transfer block
    """
    def __init__(self, in_channels, output_shape):
        print("ResnetStack")
        super(ResnetStack, self).__init__()
        # todo to be changed  =>...damn,,what to be change????
        self.conv1 = nn.Conv2d(in_channels, params.resnet_filters, kernel_size=params.generator_kernel_size, padding=(params.generator_kernel_size-1)//2)
        self.relu = nn.ReLU()
        self.resblks = self.make_layers(params.resnet_filters, params.resnet_blocks)
        self.conv2 = nn.Conv2d(params.resnet_filters, output_shape[0], kernel_size=1)
        self.tanh = nn.Tanh()

    def make_layers(self, in_channels, num_blks):
        layers = []
        for i in range(0, num_blks):
            layers.append(ResidualBlock(in_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.resblks(out)
        out = self.conv2(out)
        out = self.tanh(out)

        return out


class ResnetGenerator(nn.Module):
    """Creates a ResNet-based generator."""
    def __init__(self, source_images_size, output_shape, latent_vars=None):
        print("ResnetGenerator")
        super(ResnetGenerator, self).__init__()
        self.latent_vars = latent_vars
        in_channels = list(source_images_size)[1]
        self.resnet_stack = ResnetStack(in_channels, output_shape)

    def forward(self, x):
        # todo enable the following later
        if self.latent_vars:
            noise_channel = project_latent_vars(
                proj_shape=list(x.size())[1:3] + [1],
                latent_vars=self.latent_vars,
                combine_method='concat'
            )
            feed_x = torch.cat([x, noise_channel], 1)

        transferred_img = self.resnet_stack(x)

        return transferred_img


class ResidualInterpretationBlock(nn.Module):
    def __init__(self, images_size):
        super(ResidualInterpretationBlock, self).__init__()

        self.conv1 = nn.Conv2d(list(images_size)[1], params.res_int_filters, kernel_size=params.generator_kernel_size)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(params.res_int_filters, params.res_int_filters, kernel_size=params.generator_kernel_size)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(params.res_int_filters, 3, kernel_size=params.generator_kernel_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        residual = x
        out = self.conv2(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.tanh(out)
        # Add the residual
        out += residual
        # Clip the output
        out = torch.max(out, -1.0)
        out = torch.min(out, 1)
        return out


class ResidualInterpretationGenerator(nn.Module):
    """Creates a generator producing purely residual transformations.

    A residual generator differs from the resnet generator in that each 'block' of
    the residual generator produces a residual image. Consequently, the 'progress'
    of the model generation process can be directly observed at inference time,
    making it easier to diagnose and understand.
    """
    def __init__(self, images_size, latent_vars=None):
        super(ResidualInterpretationGenerator, self).__init__()
        self.images_size = images_size
        if latent_vars:
            projected_latent = project_latent_vars(list(images_size)[1:3] + [list(images_size)[1]],
                                                   latent_vars=latent_vars,
                                                   combine_method='sum')
            # images = torch.cat([images, projected_latent], 1)
            # todo: check why
            self.images += projected_latent

        self.res_interpretation_blks = self.make_layer(16)

    def make_layer(self):
        layers = []
        for i in range(0, params.res_int_blocks):
            layers.append(ResidualInterpretationBlock(self.images_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.res_interpretation_blks(x)


class Discriminator(nn.Module):
    """Creates a discriminator for a GAN."""
    def __init__(self, images_size):
        print("Discriminator")
        super(Discriminator, self).__init__()
        if params.discriminator_image_noise:
            # images = self.add_noise(images)
            # todo disable this
            pass

        # Due to reason specified in DCGAN, not use batch norm on discriminator input
        self.conv1 = nn.Conv2d(list(images_size)[1],
                               params.num_discriminator_filters,
                               stride=params.discriminator_first_stride,
                               kernel_size=params.discriminator_kernel_size)
        self.lrelu = nn.LeakyReLU(params.lrelu_leakiness)
        # add noise

        self.discriminator_blks = []
        block_id = 2
        in_channel = params.num_discriminator_filters
        # height after first conv layer
        height = (images_size[2] - params.discriminator_kernel_size)//params.discriminator_first_stride + 1
        # todo check how to decide the projection shape
        while height >= params.projection_shape_size:
            layers = []
            num_filters = int(params.num_discriminator_filters * (params.discriminator_filter_factor ** (block_id - 1)))
            for conv_id in range(0, params.discriminator_conv_block_size):
                layers.append(nn.Conv2d(in_channel, num_filters, kernel_size=params.discriminator_kernel_size))
                layers.append(nn.BatchNorm2d(num_filters))
                layers.append(nn.LeakyReLU(params.lrelu_leakiness))
                in_channel = num_filters
                height = (height - params.discriminator_kernel_size) + 1
            if params.discriminator_do_pooling:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
                height = (height - 2)/2+1
            #else:
             #   layers.append(
              #      nn.Conv2d(num_filters, num_filters, kernel_size=params.discriminator_kernel_size))

            # noise = self.add_noise(hidden=None, scope_num=block_id)
            # layers.append(noise)
            self.discriminator_blks.append(nn.Sequential(*layers))
            block_id += 1
        # todo: think a way to change teh hard code calculation
        self.discriminator_blks = nn.Sequential(*self.discriminator_blks)
        self.fully_connected = nn.Linear(num_filters * height * height, 1)

    # todo disable this now
    def add_noise(self, hidden, scope_num=None):
        if scope_num:
            hidden = nn.Dropout(params.discriminator_dropout_keep_prob)
        if params.discriminator_noise_stddev == 0:
            return hidden
        return hidden + torch.FloatTensor(list(hidden.size())).normal_(mean=0.0, std=params.discriminator_noise_stddev)

    def forward(self, x):
        out = self.conv1(x)
        out = self.lrelu(out)
        # while list(out.size())[2] > params.projection_shape_size:
        out = self.discriminator_blks(out)
        out = flatten(out)
        out = self.fully_connected(out)
        return out


class DoublingCNNAndQuaternion(nn.Module):
    """
    Alternate conv, pool while doubling filter count
    """
    # classifier + quaternion regressor.
    #    [conv + pool]* + FC

    def __init__(self, images_size, num_private_layers, num_classes):
        print("DoublingCNNAndQuaternion")
        super(DoublingCNNAndQuaternion, self).__init__()
        in_channel = list(images_size)[1]
        height = list(images_size)[2]
        depth = 32
        layer_id = 1

        self.private_layers = []
        while num_private_layers > 0 and height > 5:
            self.private_layers.append(self.make_layer(in_channel, depth, 3, 2, 2))
            in_channel = depth
            height = (height - 3) + 1       # conv layer
            height = (height - 2) / 2 + 1   # pooling layer
            depth *= 2
            layer_id += 1
            num_private_layers -= 1
        self.private_layers = nn.Sequential(*self.private_layers)

        self.shared_layers = []
        while height > 5:
            self.shared_layers.append(self.make_layer(in_channel, depth, 3, 2, 2))
            height = (height - 3) + 1  # conv layer
            height = (height - 2) / 2 + 1  # pooling layer
            in_channel = depth
            depth *= 2
            layer_id += 1
        self.shared_layers = nn.Sequential(*self.shared_layers)

        self.FC1 = nn.Linear(calculate_flat_fts(images_size, nn.Sequential(self.private_layers, self.shared_layers)), 100)
        self.quaternion = nn.Sequential(
            nn.Linear(100, 4),
            nn.Tanh()
        )
        self.FC2 = nn.Linear(100, num_classes)

    def make_layer(self, in_, out_, kernel_size_conv, kernel_size_pool, stride):
        return nn.Sequential(nn.Conv2d(in_, out_, kernel_size=kernel_size_conv),
                             nn.MaxPool2d(kernel_size=kernel_size_pool, stride=stride))

    def forward(self, x):
        out = x
        out = self.private_layers(out)
        out = self.shared_layers(out)
        out = flatten(out)
        out = self.FC1(out)
        out = F.dropout(out, 0.5, training=params.is_training)

        quaternion_pred = self.quaternion(out)
        quaternion_pred = F.normalize(quaternion_pred, 2, 1)

        logits = self.FC2(out)

        return logits, quaternion_pred
