import argparse
import os
import params
import construction
import losses
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import *
import itertools
from util import LoggerGenerator
from datetime import datetime


def run_training(run_dir, checkpoint_dir):
    print("**************************************************************")
    print("**************************************************************")
    print("Running from here")
    print("**************************************************************")
    print("**************************************************************")

    for path in [run_dir, checkpoint_dir]:
        if not os.path.exists(path):
            os.mkdir(path)

    # preprocess the inputs
    # todo to be changed
    preprocess = transforms.Compose([
        transforms.Scale(params.image_scale),
        transforms.ToTensor(),
        transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
    ])

    # image loading
    # create dataloader
    source_dataset = datasets.ImageFolder(root=params.source_data_path, transform=preprocess)
    source_loader = torch.utils.data.DataLoader(source_dataset,
                                                batch_size=params.batch_size, shuffle=params.shuffle_batch)
    target_dataset = datasets.ImageFolder(root=params.target_data_path, transform=preprocess)
    target_loader = torch.utils.data.DataLoader(target_dataset,
                                                batch_size=params.batch_size, shuffle=params.shuffle_batch)

    # model components
    tgt_imgs_shape = src_imgs_shape = [params.batch_size, 3, 224, 224]
    model_components = construction.create_model(target_images_shape=tgt_imgs_shape,
                                                 source_images_shape=src_imgs_shape,
                                                 num_classes=params.num_classes)
    generator = model_components['generator']
    transferred_discriminator = model_components['transferred_domain_logits']
    target_discriminator = model_components['target_domain_logits']
    source_task_classfier = model_components['source_task_classifier']
    target_task_classfier = model_components['target_task_classifier']
    transferred_task_classfier = model_components['transferred_task_classifier']

    logger.info('***** Finish creating model ******')

    # optimizers
    lr = params.learning_rate
    opt_generator = torch.optim.Adam(generator.parameters(), lr=lr, betas=params.adam_beta1)
    opt_transferred_discriminator = torch.optim.Adam(transferred_discriminator.parameters(), lr=lr, betas=params.adam_beta1)

    # for saving gradient
    grad_records = {}
    def save_grad(name):
        def hook(grad):
            grad_records[name] = grad
        return hook

    # todo implement
    exp_lr_shceduler = lr_scheduler.StepLR(opt_generator, step_size=params.lr_decay_steps, gamma=params.lr_decay_rate)

    generator_loss_records = []
    discriminator_loss_records = []

    ##################
    # Start Training #
    ##################
    for i in range(params.epochs):
        itertools.tee(source_loader)
        itertools.tee(target_loader)
        zipped_loader = enumerate(zip(source_loader, target_loader))
        avg_generator_loss = 0
        avg_discriminator_loss = 0
        logger.info("epoch {} started".format(i))
        for step, ((images_src, labels_src), (images_tgt, labels_tgt)) in zipped_loader:
            print("batch %s started" % step)
            if torch.cuda.is_available():
                images_src = images_src.cuda()
                images_tgt = images_tgt.cuda()
                labels_src = labels_src.cuda()
                labels_tgt = labels_tgt.cuda()

            # todo check whats the functionality for a Variable
            src_imgs = Variable(images_src).float()
            tgt_imgs = Variable(images_tgt).float()
            src_lbls = Variable(labels_src)
            tgt_lbls = Variable(labels_tgt)

            if tgt_imgs.dtype.name != 'float32':
                raise ValueError('target_images must be tf.float32 and [-1, 1] normalized.')

            if src_imgs.dtype.name != 'float32':
                raise ValueError('source_images must be tf.float32 and [-1, 1] normalized.')

            # specify model
            transferred_imgs = generator(src_imgs)
            transferred_discriminator = transferred_discriminator(transferred_imgs)
            transferred_pred = transferred_task_classfier(transferred_imgs)
            target_pred = target_task_classfier(tgt_imgs)

            # specify losses
            generator_loss = losses.g_step_loss(source_images=src_imgs,
                                                source_labels=src_lbls,
                                                transferred_images=transferred_imgs,
                                                transferred_domain_logits=transferred_discriminator,
                                                num_classes=params.num_classes)
            discriminator_loss = losses.d_step_loss(transfer_pred=transferred_pred,
                                                    transfer_label=src_lbls,
                                                    target_pred=target_pred,
                                                    target_label=tgt_lbls,
                                                    source_labels=src_lbls,
                                                    num_classes=params.num_classes)

            avg_generator_loss += generator_loss.data.cpu().numpy()[0]
            avg_discriminator_loss += discriminator_loss.data.cpu().numpy()[0]

            # todo backprop and update


        generator_loss_records.append(avg_generator_loss)
        discriminator_loss_records.append(avg_discriminator_loss)
        logger.info("epoch {} | g_loss {}, d_loss {}".format(i, avg_generator_loss / (step + 1),
                                                             avg_discriminator_loss / (step - 1)))

    # save model
    torch.save(generator.state_dict(), os.path.join(checkpoint_dir, 'generator.model'))
    torch.save(target_discriminator.state_dict(), os.path.join(checkpoint_dir, 'target_discriminator.model'))
    torch.save(transferred_discriminator.state_dict(), os.path.join(checkpoint_dir, 'transferred_discriminator.model'))
    # todo save classifier?
    logger.info("model saved")

    return {
        # todo return records
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', help='directory for saving outputs during running')
    parser.add_argument('--checkpoint_dir', help='directory for saving checkpoints')
    args = parser.parse_args()

    run_training(
        run_dir=args.run_dir,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == '__main__':
    current_time = str(datetime.now())[:-7].replace(" ", "T").replace(":", "-")
    log_file_name = "[{}]_{}".format(current_time, "training.txt")
    logger = LoggerGenerator.get_logger(log_file_path="log/{}".format(log_file_name))

    main()
