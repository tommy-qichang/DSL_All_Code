import argparse
import importlib

import json
import numpy as np
import os
import plotly.graph_objects as go
import torch
from plotly.offline import plot
from tqdm import tqdm

from data_loader.general_data_loader import GeneralDataset
from parse_config import ConfigParser

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    print(f'==============Configuration==============')
    print(f'{json.dumps(config._config, indent=4)}')
    print(f'==============End Configuration==============')

    logger = config.get_logger('test')

    config['data_loader'] = config['test_data_loader']
    # setup data_loader instances
    my_transform = config.init_transform(training=False)
    data_loader = config.init_obj('data_loader', transforms=my_transform, training=True)

    # build model architecture
    model = config.init_obj('model')
    logger.info(model)

    # get function handles of loss and metrics
    criterion = config.init_ftn('loss')
    metric_fns = [getattr(importlib.import_module(f"metric.{met}"), met) for met in config['metric']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    assert config.resume is not None, "In the test script, -r must be assigned to reload well-trained model."
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    save_dir = '{:s}/test_results'.format(str(config.save_dir))
    os.makedirs(save_dir, exist_ok=True)
    print(f"Save test results at {save_dir}")

    with torch.no_grad():
        for i, (data, target, misc) in enumerate(tqdm(data_loader)):
            orig_data, target = data.to(device), target.to(device)

            data = orig_data.squeeze().clone().cpu().numpy()
            tile_data, orig_split_pos, ext_shape = GeneralDataset.split_volume(data, (96, 96, 96), (48, 48, 48))

            tile_outputs = []
            for j in range(tile_data.shape[0]):
                input = torch.tensor(tile_data[j]).to(device)
                input = input.unsqueeze_(0).unsqueeze_(0)
                output = model(input)
                tile_outputs.append(output.cpu().numpy())

            tile_outputs = np.concatenate(tile_outputs, axis=0)

            outputs = []
            outputs.append(
                GeneralDataset.combine_volume(tile_outputs[:, 0], data.shape, ext_shape, orig_split_pos, (48, 48, 48)))
            outputs.append(
                GeneralDataset.combine_volume(tile_outputs[:, 1], data.shape, ext_shape, orig_split_pos, (48, 48, 48)))
            output = torch.tensor(np.stack(outputs, axis=0)).unsqueeze_(0).to(device)

            #
            # save sample images, or do something with output here
            #
            # pred = torch.argmax(output, 1)
            # pred = (pred == 1)
            # for k in range(data.size(0)):
            #     for t, m, s in zip(data[k], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]):
            #         t.mul_(s).add_(m)
            #     img_concat = torch.cat((data[k][0], target[k].float(), pred[k].float()), dim=1)
            #     save_img_name = '{:s}/{:s}'.format(save_dir, str(i))
            #     # for i, met in enumerate(metric_fns):
            #         # save_img_name += '-{:s}-{:.4f}'.format(met.__name__, total_metrics[i].item())
            #     save_image(img_concat, save_img_name+"_"+str(k)+'.png')

            if config['trainer']['vis']:
                pred = torch.argmax(output, dim=1).squeeze()
                pred = pred.cpu().numpy()
                pred = pred[:, :, ::-1]
                data = data[:, :, ::-1]
                vis_target = target.squeeze().cpu().numpy()[:, :, ::-1]

                # print(f"before edging:pred sum: {pred.sum()}")
                # pred = edging_mask(pred)
                # data = edging_mask(data)
                # vis_target = edging_mask(vis_target)
                # print(f"after edging, pred sum: {pred.sum()}")

                pred_loc = np.where(pred == 1)
                data_loc = np.where(data == 1)
                target_loc = np.where(vis_target == 1)
                # plot([
                #     go.Mesh3d(x=pred_loc[0], y=pred_loc[1], z=pred_loc[2], alphahull=1, color="lightpink", opacity=0.5),
                #     go.Mesh3d(x=target_loc[0], y=target_loc[1], z=target_loc[2], alphahull=1, color="lightpink", opacity=0.5)
                # ], filename=f'{save_dir}/3dmesh_{str(i)}.html')

                trace1 = go.Scatter3d(x=pred_loc[0], y=pred_loc[1], z=pred_loc[2], mode="markers", marker=dict(size=1),
                                      opacity=0.5, name="pred")
                trace2 = go.Scatter3d(x=data_loc[0], y=data_loc[1], z=data_loc[2], mode="markers", marker=dict(size=1),
                                      opacity=0.5, name="sparse_input")
                trace3 = go.Scatter3d(x=target_loc[0], y=target_loc[1], z=target_loc[2], mode="markers",
                                      marker=dict(size=1),
                                      opacity=0.5, name="target")
                # trace1 = go.Mesh3d(x=pred_loc[0], y=pred_loc[1], z=pred_loc[2], alphahull=1, color='lightpink',
                #                    opacity=0.5, name="pred")
                # trace2 = go.Mesh3d(x=data_loc[0], y=data_loc[1], z=data_loc[2], alphahull=1, color='red',
                #                    opacity=0.5, name="sparse_input")
                # trace3 = go.Mesh3d(x=target_loc[0], y=target_loc[1], z=target_loc[2], alphahull=1, color='blue',
                #                    opacity=0.5, name="target")
                data = [trace1, trace2, trace3]

                layout = go.Layout(
                    scene=dict(
                        xaxis=dict(nticks=5, range=[0, 120]),
                        yaxis=dict(nticks=5, range=[0, 120]),
                        zaxis=dict(nticks=5, range=[0, 120])
                    )
                )
                fig = go.Figure(data=data, layout=layout)
                plot(fig, filename=f'{save_dir}/Scatter_{str(i)}.html')

            # plot([
            #     go.Scatter3d(x=pred_loc[0], y=pred_loc[1], z=pred_loc[2], mode="markers", marker=dict(size=1),
            #                  opacity=0.5, name="pred"),
            #     go.Scatter3d(x=data_loc[0], y=data_loc[1], z=data_loc[2], mode="markers", marker=dict(size=1),
            #                  opacity=0.5, name="sparse_input"),
            #     go.Scatter3d(x=target_loc[0], y=target_loc[1], z=target_loc[2], mode="markers", marker=dict(size=1),
            #                  opacity=0.5, name="target")
            # ], filename=f'{save_dir}/Scatter3D_{str(i)}.html')
            # computing loss, metrics on test set

            loss = criterion[0](output, target)
            if len(criterion) > 1:
                for idx in range(1, len(criterion)):
                    loss += criterion[idx](output, target)

            batch_size = output.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                misc['input'] = orig_data
                total_metrics[i] += metric(output, target, misc) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


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
