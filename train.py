import argparse
import os
import params
import construction
import losses
import torch
import itertools
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import *
from logger import Logger


def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def run_training(checkpoint_dir):
    print("**************************************************************")
    print("**************************************************************")
    print("Running from here")
    print("**************************************************************")
    print("**************************************************************")

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    # preprocess the inputs
    # todo to be changed
    preprocess = transforms.Compose([
        transforms.Scale(params.image_scale),
        transforms.ToTensor(),
        transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std)
    ])

    # image loading
    # create dataloader
    print("Start loading images")
    source_dataset = datasets.ImageFolder(root=params.source_data_path, transform=preprocess)
    source_loader = torch.utils.data.DataLoader(source_dataset,
                                                batch_size=params.batch_size, shuffle=params.shuffle_batch)
    target_dataset = datasets.ImageFolder(root=params.target_data_path, transform=preprocess)
    target_loader = torch.utils.data.DataLoader(target_dataset,
                                                batch_size=params.batch_size, shuffle=params.shuffle_batch)

    print("Start creating model")
    # model components
    tgt_imgs_shape = src_imgs_shape = [params.batch_size, 3, 224, 224]
    model_components = construction.create_model(target_images_shape=tgt_imgs_shape,
                                                 source_images_shape=src_imgs_shape,
                                                 num_classes=params.num_classes)
    generator = model_components['generator']
    print(generator)
    trans_discriminator = model_components['transferred_domain_logits']
    target_discriminator = model_components['target_domain_logits']
    source_task_classifier = model_components['source_task_classifier']
    target_task_classifier = model_components['target_task_classifier']
    trans_task_classifier = model_components['transferred_task_classifier']

    # Load model
    resume = False
    if os.listdir(checkpoint_dir):
        model = torch.load(os.path.join(checkpoint_dir, 'model.pt'))
        resume_epoch = model['epoch']
        generator.load_state_dict(model['generator'])
        trans_discriminator.load_state_dict(model['trans_discriminator'])
        target_discriminator.load_state_dict(model['target_discriminator'])
        source_task_classifier.load_state_dict(model['source_task_classifier'])
        target_task_classifier.load_state_dict(model['target_task_classifier'])
        trans_task_classifier.load_state_dict(model['trans_task_classifier'])
        resume = True

    print('***** Finish creating model ******')

    # optimizers
    lr = params.learning_rate
    opt_generator = torch.optim.Adam(generator.parameters(), lr=lr, betas=params.adam_betas)
    opt_trans_discriminator = torch.optim.Adam(trans_discriminator.parameters(), lr=lr, betas=params.adam_betas)
    opt_target_discriminator = torch.optim.Adam(target_discriminator.parameters(), lr=lr, betas=params.adam_betas)
    opt_trans_classifier = torch.optim.Adam(trans_task_classifier.parameters(), lr=lr, betas=params.adam_betas)
    opt_target_classifier = torch.optim.Adam(target_task_classifier.parameters(), lr=lr, betas=params.adam_betas)
    opt_source_classifier = torch.optim.Adam(source_task_classifier.parameters(), lr=lr, betas=params.adam_betas)

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

    if resume:
        start_epoch = resume_epoch
    else:
        start_epoch = 0
        
    ##################
    # Start Training #
    ##################
    for i in range(start_epoch, params.epochs):
        itertools.tee(source_loader)
        itertools.tee(target_loader)
        zipped_loader = enumerate(zip(source_loader, target_loader))
        avg_generator_loss = 0
        avg_discriminator_loss = 0
        print("********** epoch %d started **********" % i)
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

            # specify model
            trans_imgs = generator(src_imgs)
            trans_domain_logits = trans_discriminator(trans_imgs)
            target_domain_logits = target_discriminator(tgt_imgs)
            source_task_logits, source_quaternion = source_task_classifier(src_imgs)
            trans_task_logits, transfer_quaternion = trans_task_classifier(trans_imgs)
            target_task_logits, target_quaternion = target_task_classifier(tgt_imgs)

            # specify losses
            generator_loss = losses.g_step_loss(source_images=src_imgs,
                                                source_labels=src_lbls,
                                                source_task_logits=source_task_logits,
                                                trans_images=trans_imgs,
                                                trans_task_logits=trans_task_logits,
                                                trans_domain_logits=trans_domain_logits)

            discriminator_loss = losses.d_step_loss(trans_task_logits=trans_task_logits,
                                                    trans_domain_logits=trans_domain_logits,
                                                    target_domain_logits=target_domain_logits,
                                                    source_labels=src_lbls,
                                                    source_task_logits=source_task_logits)

            avg_generator_loss += generator_loss.data.cpu().numpy()[0]
            avg_discriminator_loss += discriminator_loss.data.cpu().numpy()[0]

            # forward, backprop and updating
            opt_generator.zero_grad()
            opt_target_discriminator.zero_grad()
            opt_trans_discriminator.zero_grad()
            opt_target_classifier.zero_grad()
            opt_trans_classifier.zero_grad()
            opt_source_classifier.zero_grad()

            generator_loss.backward(retain_graph=True)
            discriminator_loss.backward(retain_graph=True)

            opt_generator.step()
            opt_target_discriminator.step()
            opt_trans_discriminator.step()
            opt_target_classifier.step()
            opt_trans_classifier.step()
            opt_source_classifier.step()

            s = src_imgs[0]
            t = tgt_imgs[0]
            tr = trans_imgs[0]

            for count in range(3):
                utils.save_image(images_src[count], 'images/images_src' + str(count) + '.jpg', 'JPEG')
                utils.save_image(src_imgs.data[count], 'images/s' + str(count) + '.jpg', 'JPEG')
                utils.save_image(tgt_imgs.data[count], 'images/t' + str(count) + '.jpg', 'JPEG')
                utils.save_image(trans_imgs.data[count], 'images/tr' + str(count) + '.jpg', 'JPEG')
        generator_loss_records.append(avg_generator_loss)
        discriminator_loss_records.append(avg_discriminator_loss)
        print("epoch %d | g_loss %f, d_loss %f" % (i, avg_generator_loss / (step + 1),
                                                   avg_discriminator_loss / (step - 1)))

        # ---- Tensorboard Logging ---- #
        # (1) Log the scalar values
        info = {
            'avg_generator_loss': avg_generator_loss,
            'avg_discriminator_loss': avg_discriminator_loss
        }
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step)
        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in generator.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, to_np(value), step)
            logger.histo_summary(tag + '/grad', to_np(value.grad), step)
        for tag, value in trans_discriminator.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, to_np(value), step)
            logger.histo_summary(tag + '/grad', to_np(value.grad), step)
        # (3) Log the images
        info = {
            'source_images': to_np(src_imgs.view(-1, 224, 224)[:10]),
            'transfer_images': to_np(trans_imgs.view(-1, 224, 224)[:10]),
            'target_images': to_np(tgt_imgs.view(-1, 224, 224)[:10])
        }
        for tag, images in info.items():
            logger.image_summary(tag, images, step)

        if i % 10 == 0:
            # save model
            torch.save({
                'epoch': i,
                'generator': generator.state_dict(),
                'target_discriminator': target_discriminator.state_dict(),
                'trans_discriminator': trans_discriminator.state_dict(),
                'source_task_classifier': source_task_classifier.state_dict(),
                'target_task_classifier': target_task_classifier.state_dict(),
                'trans_task_classifier': trans_task_classifier.state_dict()
            }, os.path.join(checkpoint_dir, 'model.pt'))
            print("model saved")

    return {
        # todo return records
    }


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--checkpoint_dir', help='directory for saving checkpoints')
    # args = parser.parse_args()

    run_training(
        checkpoint_dir="checkpoint_dir/",
    )


if __name__ == '__main__':
    logger = Logger('./log')
    main()
