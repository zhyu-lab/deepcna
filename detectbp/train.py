import argparse
import os
import datetime
import torch
import numpy as np
from matplotlib import pyplot as plt
from cbseg import segment
import autoencoder
from autoencoder import AE
from genomedata import GenomeData

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(args):
    start_t = datetime.datetime.now()

    output_dir = args.output
    if not os.path.isdir(output_dir):
        os.mkdir(args.output)

    gd = GenomeData(args.input)
    gd.load_data()
    gd.preprocess_data()

    data = gd.data_lrc_all.copy()
    data_bk = data.copy()
    print(data.shape)

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    setup_seed(args.seed)

    # define and create AE architecture
    ae_model = AE(data.shape[1], args.latent_dim).cuda()

    epochs = []
    train_loss = []
    test_acc = []

    optimizer = torch.optim.Adam(ae_model.parameters(), lr=args.lr)

	# Start training the model
    ae_model.train()
    for epoch in range(args.epochs):
        print("epoch", epoch)
        epochs.append(epoch)
        total_loss = 0
        for step, x in autoencoder.xs_gen(data, args.batch_size, 1):
            x = torch.from_numpy(x).cuda()
            x = x.squeeze()
            z, y = ae_model(x)

            loss = autoencoder.mse_loss(y, x)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("epoch:", epoch, "| train loss: %.4f" % total_loss.data.cpu().numpy())
        train_loss.append(total_loss.data.cpu().numpy())

    output_dir = args.output
    if not os.path.isdir(output_dir):
        os.mkdir(args.output)
    ll_file = output_dir + '/loss.txt'
    if os.path.isfile(ll_file):
        os.remove(ll_file)
    file_o = open(ll_file, 'w')
    np.savetxt(file_o, np.c_[np.reshape(train_loss, (1, len(train_loss)))], fmt='%f', delimiter=',')
    file_o.close()

    # get latent representation of single cells after CAE training is completed
    z_hidden = []
    x_cst = []
    data = data_bk.copy()

    ae_model.eval()
    for step, x in autoencoder.xs_gen(data, args.batch_size, 0):
        x = torch.from_numpy(x).cuda()
        x = x.squeeze()
        with torch.no_grad():
            z, y = ae_model(x)
            z = z.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            z_hidden.append(z)
            x_cst.append(y)

    data_hidden = []
    for i, z in enumerate(z_hidden):
        if i == 0:
            data_hidden = z
        else:
            data_hidden = np.r_[data_hidden, z]
    data_hidden = np.double(data_hidden)

    data_pos_all = (gd.data_bin_all - 1) * gd.bin_size
    max_value = np.max(data_hidden)
    min_value = np.min(data_hidden)

    # use CBS to segment the genome
    chromosomes = np.unique(gd.data_chr_all)
    breakpoints_all = []
    for i in range(len(chromosomes)):
        tv = gd.data_chr_all == chromosomes[i]
        data_pos = data_pos_all[tv]
        breakpoints_chr = []
        for j in range(data_hidden.shape[1]):
            y = data_hidden[tv, j]
            plt.clf()
            plt.plot(data_pos, y, '.b', markersize=5)
            segments = segment(y, shuffles=1000, p=0.1)
            for seg in segments:
                print(seg)
                s_index = seg.start
                e_index = seg.end
                breakpoints_chr.append(e_index)
                mean_value = np.median(y[s_index:e_index])
                plt.plot([data_pos[s_index], data_pos[e_index - 1]], [mean_value, mean_value], '-r')
            plt.axis([min(data_pos), max(data_pos), min_value - 5, max_value + 5])
            plt.title('Chromosome_' + str(chromosomes[i]) + '_' + str(j))
            plt.xlabel('Position')
            plt.ylabel('Latent feature')
            fig_path = output_dir + '/chr_' + str(chromosomes[i]) + '_' + str(j) + '.png'
            plt.savefig(fig_path, format='png', dpi=200)
            plt.show()

        breakpoints_chr = np.unique(breakpoints_chr)
        tmp = [breakpoints_chr[0]]
        for j in np.arange(1, len(breakpoints_chr), 1):
            if breakpoints_chr[j]-tmp[-1] < args.min_size:
                tmp[-1] = int((breakpoints_chr[j]+tmp[-1])/2)
            else:
                tmp.append(breakpoints_chr[j])
        breakpoints_all.append(tmp)

    print('inferred breakpoints:', breakpoints_all)

    lrc_file = output_dir + '/lrc.txt'
    if os.path.isfile(lrc_file):
        os.remove(lrc_file)
    file_o = open(lrc_file, 'a')
    np.savetxt(file_o, np.c_[gd.bin_size], fmt='%d', delimiter=',')
    np.savetxt(file_o, np.c_[np.reshape(gd.data_chr_all, (1, len(gd.data_chr_all)))], fmt='%d', delimiter=',')
    np.savetxt(file_o, np.c_[np.reshape(gd.data_bin_all, (1, len(gd.data_bin_all)))], fmt='%d', delimiter=',')
    np.savetxt(file_o, np.c_[np.transpose(gd.data_lrc_all)], fmt='%.3f', delimiter=',')
    file_o.close()

    barcode_file = output_dir + '/barcode.txt'
    file_o = open(barcode_file, 'w')
    np.savetxt(file_o, np.c_[np.reshape(gd.barcodes, (1, len(gd.barcodes)))], fmt='%s', delimiter=',')
    file_o.close()

    seg_file = output_dir + '/seg.txt'
    file_o = open(seg_file, 'w')
    file_o.write('chr\tstart\tend\n')
    for i in range(len(chromosomes)):
        breakpoints_chr = breakpoints_all[i]
        s_index = 1
        for e_index in breakpoints_chr:
            file_o.write('{}\t{}\t{}\n'.format(chromosomes[i], s_index, e_index))
            s_index = e_index+1
    file_o.close()

    latent_file = output_dir + '/latent.txt'
    file_o = open(latent_file, 'w')
    for j in range(data_hidden.shape[1]):
        features = data_hidden[:, j]
        for i in range(len(features)-1):
            file_o.write('{},'.format(features[i]))
        file_o.write('{}\n'.format(features[-1]))
    file_o.close()

    end_t = datetime.datetime.now()
    print('elapsed time: ', (end_t-start_t).seconds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepCNA")
    parser.add_argument('--epochs', type=int, default=500, help='number of epoches to train the AE.')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate.')
    parser.add_argument('--latent_dim', type=int, default=2, help='latent dimensionality.')
    parser.add_argument('--min_size', type=int, default=3, help='minimum size of segments.')
    parser.add_argument('--seed', type=int, default=0, help='seed to train the model.')
    parser.add_argument('--input', type=str, default='', help='a file containing read counts, GC-content and mappability data.')
    parser.add_argument('--output', type=str, default='', help='a directory to save results.')
    args = parser.parse_args()
    main(args)
