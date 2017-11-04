import argparse
import os

def check_dirs(dirs):
    dirs = [dirs] if type(dirs) not in [list, tuple] else dirs
    for d in dirs:
        try:
            os.makedirs(d)
        except OSError:
            pass
    return

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sourceDataset', required=True, help='mnist | cifar10 | mnistm | usps')
    parser.add_argument('--targetDataset', required=True, help='mnist | cifar10 | mnistm | usps')
    parser.add_argument('--sourceroot', required=True, help='path to source dataset')
    parser.add_argument('--targetroot', default='.', help='path to target dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=10, help='size of the latent z vector')
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--manualSeed', type=int, default=9926, help='manual seed')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

    ## Saved model and images and checkpoint paths ##
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--netT', default='', help="path to netT (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images')
    parser.add_argument('--chkpt', default='checkpoint', help='folder to save model checkpoints')

    ## Discrminator hyper-parameters ##
    parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters')
    parser.add_argument('--leakiness', type=float, default=0.2, help='leaky relu leakiness')
    parser.add_argument('--D_conv_block_size', type=int, default=1, help='discriminator conv block size')
    parser.add_argument('--D_projection_size', type=int, default=4, help='discriminator image size after conv layers')
    parser.add_argument('--D_keep_prob', type=float, default=0.9, help='dropout keep probability')
    parser.add_argument('--D_noise_mean', type=float, default=0.0, help='discriminator external noise mean')
    parser.add_argument('--D_noise_stddev', type=float, default=0.2, help='discriminator external noise stddev')

    ## Generator hyper-parameters ##
    parser.add_argument('--ngf', type=int, default=64, help='number of generator filters')
    parser.add_argument('--G_residual_blocks', type=int, default=6, help='generator number of residual blocks')
    parser.add_argument('--G_noise_channels', type=int, default=1, help='generator number of noise channels')
    parser.add_argument('--G_noise_dim', type=int, default=10, help='generator noise dimension')

    ## Loss Weights and learning rates ##
    parser.add_argument('--style_transfer_loss_wt', type=float, default=1.0, help='generator loss weight due to discriminator')
    parser.add_argument('--domain_loss_wt', type=float, default=1.0, help='discriminator loss weight')
    parser.add_argument('--G_task_loss_wt', type=float, default=1.0, help='classifier loss weight in generator')
    parser.add_argument('--D_task_loss_wt', type=float, default=1.0, help='classifier loss weight in discriminator')
    parser.add_argument('--task_loss_wt', type=float, default=1.0, help='classifier loss weight')
    parser.add_argument('--lr_gan', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--lr_clf', type=float, default=0.001, help='learning rate, default=0.001')
    parser.add_argument('--lr_decay_rate', type=float, default=0.95, help='learning rate decay rate')
    parser.add_argument('--lr_decay_step', type=int, default=20000, help='learning rate decay step')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2 weight decay')


    opt = parser.parse_args()
    check_dirs([opt.outf, opt.chkpt])
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)

    return opt
