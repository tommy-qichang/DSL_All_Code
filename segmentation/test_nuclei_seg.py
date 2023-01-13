import argparse
import torch
import os, random
import numpy as np
# from torchvision.utils import save_image
import skimage.morphology as morph
from skimage import measure, io
# from tqdm import tqdm
# import data_loader.data_loaders as module_data
# import loss.loss as module_loss
import metric.metric_nuclei_seg as module_metric
# from metric.metric_nuclei_seg import aji
# import model.unet as module_arch
from parse_config import ConfigParser


def split_forward(model, input, size, overlap, outchannel=3):
    '''
    split the input image for forward process
    '''

    b, c, h0, w0 = input.size()

    # zero pad for border patches
    pad_h = 0
    if h0 - size > 0:
        pad_h = (size - overlap) - (h0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, pad_h, w0)).cuda()
        input = torch.cat((input, tmp), dim=2)

    if w0 - size > 0:
        pad_w = (size - overlap) - (w0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, h0 + pad_h, pad_w)).cuda()
        input = torch.cat((input, tmp), dim=3)

    _, c, h, w = input.size()

    output = torch.zeros((input.size(0), outchannel, h, w)).cuda()
    for i in range(0, h - overlap, size - overlap):
        r_end = i + size if i + size < h else h
        ind1_s = i + overlap // 2 if i > 0 else 0
        ind1_e = i + size - overlap // 2 if i + size < h else h
        for j in range(0, w - overlap, size - overlap):
            c_end = j + size if j + size < w else w

            input_patch = input[:, :, i:r_end, j:c_end]
            input_var = input_patch
            with torch.no_grad():
                output_patch = model(input_var)

            ind2_s = j + overlap // 2 if j > 0 else 0
            ind2_e = j + size - overlap // 2 if j + size < w else w
            output[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = output_patch[:, :, ind1_s - i:ind1_e - i,
                                                         ind2_s - j:ind2_e - j]

    output = output[:, :, :h0, :w0]

    return output


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = config.init_obj('test_data_loader')
    # data_loader = getattr(module_data, config['test_data_loader']['type'])(
    #     config['test_data_loader']['args']['h5_filepath'],
    #     batch_size=1, #config['test_data_loader']['args']['batch_size'],
    #     shuffle=False,
    #     validation_split=0.0,
    #     training=False,
    #     num_workers=2
    # )

    # build model architecture
    model = config.init_obj('model')
    # logger.info(model)

    # get function handles of  metrics
    test_metric_names = config['test_metrics']
    metric_fns = [getattr(module_metric, met) for met in test_metric_names]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # total_metrics = torch.zeros(len(metric_fns))
    total_metrics = []
    for i in range(len(metric_fns)):
        total_metrics.append([])

    save_dir = '{}/test_results'.format(str(config.save_dir))
    logger.info('save to ' + save_dir)
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for i, (data, _, _, instance_label, img_names) in enumerate(data_loader):
            data = data.to(device)
            output = split_forward(model, data, 224, 80)
            # print(data.unique())
            # print(target.unique())

            batch_size = data.shape[0]

            pred = torch.argmax(output, dim=1).detach().cpu().numpy()
            pred_inside = pred == 1
            target = instance_label.detach().cpu().numpy()
            for k in range(batch_size):
                pred_inside[k] = morph.remove_small_objects(pred_inside[k], 20)  # remove small object
                pred[k] = measure.label(pred_inside[k])  # connected component labeling
                pred[k] = morph.dilation(pred[k], selem=morph.selem.disk(2))
                target[k] = measure.label(target[k])

            for k, metric in enumerate(metric_fns):
                # total_metrics[k] += metric(pred, target, istrain=False) * batch_size
                total_metrics[k].append(float(metric(pred, target, istrain=False).item()))

            # pred = torch.argmax(output, dim=1)
            # print("Saving image results...")
            for k in range(data.size(0)):
                for t, m, s in zip(data[k], (0.7442, 0.5381, 0.6650), (0.1580, 0.1969, 0.1504)):
                    t.mul_(s).add_(m)
                image = data[k].permute(1, 2, 0).cpu().numpy()
                pred_colored = np.zeros(image.shape)
                target_colored = np.zeros(image.shape)
                for n in range(1, pred[k].max() + 1):
                    pred_colored[pred[k] == n, :] = np.array(get_random_color())
                for n in range(1, target[k].max() + 1):
                    target_colored[target[k] == n, :] = np.array(get_random_color())
                img_concat = np.concatenate([image, target_colored, pred_colored], axis=1)
                save_img_name = '{}/{}.png'.format(save_dir, img_names[k])
                io.imsave(save_img_name, img_concat)

    n_samples = len(data_loader.sampler)
    # print(total_metrics)
    # result_dict = {met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)}
    result_dict_mean = {met.__name__: np.mean(total_metrics[i]) for i, met in enumerate(metric_fns)}
    result_dict_std = {met.__name__: np.std(total_metrics[i]) for i, met in enumerate(metric_fns)}

    # log.update(result_dict)

    logger.info('accuracy: {:.4f} ({:.4f}), dice: {:.4f} ({:.4f}), aji: {:.4f} ({:.4f})'
                .format(result_dict_mean['accuracy'], result_dict_std['accuracy'], result_dict_mean['dice'], result_dict_std['dice'], result_dict_mean['aji'], result_dict_std['aji']))
    # Save
    result_dict = {met.__name__: total_metrics[i] for i, met in enumerate(metric_fns)}
    np.save(os.path.join(save_dir, 'metrics.npy'), result_dict)

    # Load
    # read_dictionary = np.load(os.path.join(save_dir, 'metrics.npy'), allow_pickle='TRUE').item()
    # print(read_dictionary)


def get_random_color():
    ''' generate rgb using a list comprehension '''
    r, g, b = [random.random() for i in range(3)]
    return r, g, b


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
