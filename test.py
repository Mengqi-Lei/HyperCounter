import torch
import os
import argparse
from datasets.crowd_rgbtcc import Crowd_RGBTCC
from datasets.crowd_drone import Crowd_Drone

from models.HyperCroM import fusion_model
from utils.evaluation import eval_game, eval_relative
from PIL import Image


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--dataset', default='RGBTCC', choices=['RGBTCC', 'DroneRGBT'],
                    help='the dataset to train, RGBTCC or DroneRGBT')
parser.add_argument('--data-dir', default='data/RGBT-CC/RGBT-CC-use',
                        help='training data directory')
parser.add_argument('--model-path', default=''
                    , help='model name')


# model setting
parser.add_argument('--constr-hg', type=str, default='threshold', choices=['threshold', 'knn'],
                    help='the availabel methods for cross-modal hypergraph construction')
parser.add_argument('--constr-k', type=int, default=4,
                    help='the k for knn method')
parser.add_argument('--constr-threshold', type=float, default=0.8,
                    help='the threshold for threshold method, 0.8 is recommended for RGBTCC, 0.5 is recommended for DroneRGBT')


parser.add_argument('--device', default='0', help='gpu device')
args = parser.parse_args()

if __name__ == '__main__':

    if args.dataset == 'RGBTCC':
        datasets = Crowd_RGBTCC(os.path.join(args.data_dir, 'test'), method='test')
        dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                                num_workers=8, pin_memory=True)
    elif args.dataset == 'DroneRGBT':
        datasets = Crowd_Drone(os.path.join(args.data_dir, 'test'), method='test')
        dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                                num_workers=8, pin_memory=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device('cuda')

    model = fusion_model(
            construct_method=args.constr_hg,
            k_interact=args.constr_k,
            thresh_interact=args.constr_threshold
        )
    
    model.to(device)
    model_path = args.model_path
    checkpoint = torch.load(model_path, device)
    model.load_state_dict(checkpoint)
    model.eval()

    print('testing...')
    # Iterate over data.
    game = [0, 0, 0, 0]
    mse = [0, 0, 0, 0]
    total_relative_error = 0

    for inputs, target, name in dataloader:
        if type(inputs) == list:
            inputs[0] = inputs[0].to(device)
            inputs[1] = inputs[1].to(device)
        else:
            inputs = inputs.to(device)

        # inputs are images with different sizes
        if type(inputs) == list:
            assert inputs[0].size(0) == 1
        else:
            assert inputs.size(0) == 1, 'the batch size should equal to 1 in validation mode'
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            # outputs,_,_,_ = outputs
            for L in range(4):
                abs_error, square_error = eval_game(outputs, target, L)
                game[L] += abs_error
                mse[L] += square_error
            relative_error = eval_relative(outputs, target)
            total_relative_error += relative_error

            vis_path = os.path.join('vis/'+ args.dataset)
            if not os.path.exists(vis_path):
                os.makedirs(vis_path)
            path = os.path.join(vis_path, name[0] + '.png')
            img =outputs.data.cpu().numpy()
            img = img.squeeze(0).squeeze(0)
            print(name[0],img.sum())
            img = img * 255.
            img = Image.fromarray(img).convert('L')
            img.save(path)


    N = len(dataloader)
    game = [m / N for m in game]
    mse = [torch.sqrt(m / N) for m in mse]
    total_relative_error = total_relative_error / N

    log_str = 'Test{}, GAME0 {game0:.2f} GAME1 {game1:.2f} GAME2 {game2:.2f} GAME3 {game3:.2f} ' \
              'MSE {mse:.2f} Re {relative:.4f}, '.\
        format(N, game0=game[0], game1=game[1], game2=game[2], game3=game[3], mse=mse[0], relative=total_relative_error)

    print(log_str)

