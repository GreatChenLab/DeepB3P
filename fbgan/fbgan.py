# -- coding: utf-8 --
# author : TangQiang
# time   : 2023/8/8
# email  : tangqiang.0701@gmail.com
# file   : fbgan.py

import os
import shutil
import glob

from tqdm import tqdm
from torch import autograd

from fbgan.models import *
from fbgan.predictor.analyser import run_analyser
from utils.amino_acid import *
from utils.utils import *


class FBGANBBB():
    def __init__(self, params, logger):
        super(FBGANBBB, self).__init__()
        self.is_blast = params.is_blast
        self.is_predict = params.is_predict
        self.n_sequences = params.n_sequences
        self.load_old = params.load_old
        self.hidden = params.hidden
        self.n_head = params.n_heads
        self.batch_size = params.batch_size
        self.lr = params.lr
        self.n_epochs = params.n_epochs
        self.seq_len = params.seq_len
        self.d_steps = params.d_steps
        self.cutoff = params.cutoff
        self.blast_path = params.blast_path
        self.predictor_path = params.predictor_path
        self.lamda = 10  # lambda
        self.n_chars = params.vocab_size
        self.device = params.device
        self.out_dir = params.out_dir
        self.sample_dir = params.sample_dir
        self.checkpoint_dir = params.checkpoint
        self.logger = logger
        self.build_model()

    def build_model(self):
        self.logger.info('begin build model with\nepoch {0}\nhidden {1}\nseq_len {2}\n'
                     'n_chars {3}\nbatch_size {4}\nblast is {5}\npredict is {6}'.format(
            self.n_epochs, self.hidden, self.seq_len, self.n_chars, self.batch_size, self.is_blast, self.is_predict)
        )
        self.G = Generator(n_chars=self.n_chars, seq_len=self.seq_len,
                           bs=self.batch_size, hidden=self.hidden).to(self.device)
        self.D = Discriminator(n_chars=self.n_chars, seq_len=self.seq_len,
                               hidden=self.hidden, n_head=self.n_head).to(self.device)
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.criterion = torch.nn.BCELoss()
        if self.load_old is not True:
            if os.path.exists(self.out_dir): shutil.rmtree(self.out_dir)
            if os.path.exists(self.sample_dir): shutil.rmtree(self.sample_dir)
            if os.path.exists(self.checkpoint_dir): shutil.rmtree(self.checkpoint_dir)
        if not os.path.exists(self.out_dir): os.makedirs(self.out_dir)
        if not os.path.exists(self.sample_dir): os.makedirs(self.sample_dir)
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)

    def save_model(self, epoch):
        torch.save(self.G.state_dict(), os.path.join(self.checkpoint_dir, f"G_weights_{epoch}.pth"))
        torch.save(self.D.state_dict(), os.path.join(self.checkpoint_dir, f"D_weights_{epoch}.pth"))

    def remove_old_indices(self, numToAdd):
        self.logger.info(f'remove {numToAdd} seqs')
        toRemove = np.argsort(self.labels)[-numToAdd:]
        self.x_data = np.delete(self.x_data, toRemove, axis=0)
        self.labels = np.delete(self.labels, toRemove)

    def load_model(self, directory=None, iteration=None):
        epoch_found = 0
        if directory is None:
            directory = self.checkpoint_dir
        list_G = glob.glob(os.path.join(directory, "G*.pth"))
        list_D = glob.glob(os.path.join(directory, "D*.pth"))
        if len(list_G) == 0:
            self.logger.warning("Checkpoint not found! Starting from scratch.")
            return 0  # file is not there
        self.logger.info("Find the most recently saved...")
        G_max_file = max(list_G, key=lambda x: int((x.split('_')[-1]).split('.')[0]))
        D_max_file = max(list_D, key=lambda x: int((x.split('_')[-1]).split('.')[0]))
        G_epoch_found = int((G_max_file.split('_')[-1]).split('.')[0])
        D_epoch_found = int((D_max_file.split('_')[-1]).split('.')[0])
        if G_epoch_found != D_epoch_found:
            self.logger.warning("Checkpoint not found! Starting from scratch.")
            return -1  # G and D not match
        max_epoch_trained = G_epoch_found
        print(max_epoch_trained, iteration)
        if iteration is None:
            self.logger.info("Loading most recently saved...")
            G_file = G_max_file
            D_file = D_max_file
        else:
            if max_epoch_trained >= iteration:
                G_file = os.path.join(directory,"G_weights_{}.pth".format(iteration))
                D_file = os.path.join(directory,"D_weights_{}.pth".format(iteration))
                epoch_found = iteration
            else:
                G_file = G_max_file
                D_file = D_max_file
                epoch_found = max_epoch_trained
        if os.path.isfile(G_file) and os.path.isfile(D_file):
            self.logger.info("Checkpoint {} found at {}!".format(epoch_found, directory))
            self.G.load_state_dict(torch.load(G_file))
            self.D.load_state_dict(torch.load(D_file))
        else:
            self.logger.info("Checkpoint file:\n {}\n{} \nnot found".format(G_file, D_file))
            self.logger.info("begin train fbgan model from {} epochs".format(0))
            epoch_found = 0
        return epoch_found

    def calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(self.batch_size, 1, 1).to(self.device)
        alpha = alpha.expand_as(real_data)  # [N,seq_len,n_chars]
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)  # [N,seq_len,n_chars]
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.D(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.contiguous().view(self.batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = self.lamda * ((gradients_norm - 1) ** 2).mean()
        return gradient_penalty

    # def sample_generator(self, num_sample, seed):
    #     torch.random.manual_seed(seed)
    #     z_input = autograd.Variable(torch.randn(num_sample, self.hidden)).to(self.device)
    #     generated_data = self.G(z_input)
    #     return generated_data

    def sample_generator(self, num_sample):
        z_input = autograd.Variable(torch.randn(num_sample, self.hidden)).to(self.device)
        generated_data = self.G(z_input)
        return generated_data

    def sample(self, num_batches_sample):
        decoded_seqs = []
        for i in range(num_batches_sample):
            z = autograd.Variable(torch.randn(self.batch_size, 128)).to(self.device)
            self.G.eval()
            generation = self.G(z)  # [batch,seq_len,n_chars]
            seqs = (generation.data).cpu().numpy()
            hard = np.argmax(seqs, axis=2)  # [N,seq_len]
            pseudo = [from_id_to_amino_acid(one) for one in hard]
            decoded_seqs.extend(pseudo)
        self.G.train()
        return decoded_seqs

    def generator_train(self):
        self.G_optimizer.zero_grad()
        g_fake_data = self.sample_generator(self.batch_size)
        dg_fake_pred = self.D(g_fake_data)
        g_error_total = -torch.mean(dg_fake_pred)
        g_error_total.backward()
        self.G_optimizer.step()
        return g_error_total.detach().cpu().numpy()

    def discriminator_train(self, real_data):
        self.D_optimizer.zero_grad()

        real_data = real_data.to(self.device)
        d_real_pred = self.D(real_data)
        d_real_error = torch.mean(d_real_pred)

        fake_data = self.sample_generator(self.batch_size)
        d_fake_pred = self.D(fake_data)
        d_fake_error = torch.mean(d_fake_pred)

        gradient_penalty = self.calc_gradient_penalty(real_data, fake_data)

        d_error_total = d_fake_error - d_real_error + gradient_penalty
        w_dist = d_real_error - d_fake_error
        d_error_total.backward()
        self.D_optimizer.step()

        return d_fake_error.detach().cpu().numpy(), d_real_error.detach().cpu().numpy(), \
               gradient_penalty.detach().cpu().numpy(), d_error_total.detach().cpu().numpy(), w_dist.detach().cpu().numpy()

    def analyser(self, n_bs_samples, epoch):
        pseudo = []
        sampled_seqs = self.sample(n_bs_samples)
        pseudo.extend(remove_pad_from_str_seq(one, '0') for one in sampled_seqs)
        pseudo = list(filter(None, pseudo))
        self.logger.info(f'pseudo samples: \n {pseudo}')
        preds = run_analyser(pseudo, self.logger, self.blast_path, self.predictor_path, self.is_blast, self.is_predict)  # df ID, seq, prob1, prob2, prob3, mean
        pos_seqs = []
        if preds.shape[0] != 0:
            preds.to_csv(self.sample_dir / "sampled_{}_preds.txt".format(epoch), index=None)
            pos_seqs = preds[(preds.means >= self.cutoff) & (preds.prob_BBBPpred >= self.cutoff) &
                             (preds.prob_b3pred >= self.cutoff) & (preds.prob_BBBPpredict >= self.cutoff)
                             ].peptides.tolist()
            self.logger.info(f'Add {len(pos_seqs)} positive sequences for epoch {epoch}')
            with open(self.checkpoint_dir / "positives.txt", 'a+') as f:
                f.write(
                    f"Epoch: {epoch} \t Pos: {len(pos_seqs)} \t Pseudo: {len(pseudo)} \t Sampled_seqs: {len(sampled_seqs)}\n")

            if pos_seqs:
                #self.remove_old_indices(len(pos_seqs))
                new_data = [np.asarray(from_amino_acid_to_id(one.ljust(self.seq_len, '0'))) for one in pos_seqs]
                new_data = np.stack(new_data, axis=0).astype(np.int64)
                self.x_data = np.concatenate([self.x_data, new_data], axis=0)
                # the label of real data is 0
                self.labels = np.concatenate([self.labels, np.repeat(epoch, len(pos_seqs))])
                perm = np.random.permutation(len(self.x_data))
                self.x_data = self.x_data[perm]
                self.labels = self.labels[perm]
        self.logger.info(f"the training data num is {len(self.x_data)}")
        return pos_seqs

    def gan_sequences(self):
        self.logger.info(f"gan sequences {self.n_sequences}")
        # generated sequence
        total = []
        #seed = 2022 #this is for test model, the same seed with same results
        while True:
            #generation = self.sample_generator(self.batch_size, seed).detach().cpu().numpy()  # [N,seq_len,n_chars]
            #seed += 1
            generation = self.sample_generator(self.batch_size).detach().cpu().numpy()  # [N,seq_len,n_chars]
            hard = np.argmax(generation, axis=2)  # [N,seq_len]
            pseudo = [from_id_to_amino_acid(data) for data in hard]
            pseudo = [remove_pad_from_str_seq(one, '0') for one in pseudo]
            pseudo = list(filter(None, pseudo))
            total.extend(pseudo)
            total = list(set(total))
            if len(total) > self.n_sequences:
                break
        random.seed(2022)
        total.sort()
        seqs = random.sample(total, self.n_sequences)
        df = pd.DataFrame({'peptide': seqs})
        df.to_csv(os.path.join(self.out_dir, f'df_all_final_{self.n_epochs}.csv'), index=None)
        return None


    def train_model(self, x_data, y_data):
        old_epoch = 0
        loss_file = self.out_dir / 'loss.txt'
        if self.load_old:
            loss_f = open(loss_file, 'a')
            old_epoch = self.load_model(iteration=self.n_epochs)
            if old_epoch >= self.n_epochs:
                self.logger.error("The model is already trained")
                if self.n_sequences > 0:
                    self.gan_sequences()
                return None
        else:
            loss_f = open(loss_file, 'w')
        loss_f.write('d_real_loss,d_fake_loss,D_loss,G_loss,W_dist\n')
        self.x_data = x_data
        self.labels = np.zeros(len(self.x_data))  # this marks at which epoch this data was added
        num_batches_sample = 10
        is_analyser = True if self.is_blast or self.is_predict else False
        for epoch in range(old_epoch + 1, self.n_epochs + 1):
            pos_seqs = []
            if epoch % 2 == 0: self.save_model(epoch)
            d_fake_losses, d_real_losses, grad_penalties = [], [], []
            G_losses, D_losses, W_dist = [], [], []

            # if epoch > self.n_epochs * 0.1 and is_analyser:
            if epoch > 20 and is_analyser:
                pos_seqs = self.analyser(num_batches_sample, epoch)
            dataset = SeqDataset(x=self.x_data, y=self.labels, keep_one_hot=True)
            real_dataloader = get_dataloader(dataset, self.batch_size, shuffle=True, drop_last=True)
            for features, _ in tqdm(real_dataloader, mininterval=1, desc='Train Processing', leave=False):
                for p in self.D.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update
                for _ in range(self.d_steps):  # Train D
                    d_fake_err, d_real_err, gradient_penalty, d_error_total, w_dist = self.discriminator_train(features)
                d_fake_losses.append(d_fake_err)
                d_real_losses.append(d_real_err)
                grad_penalties.append(gradient_penalty)
                D_losses.append(d_error_total)
                W_dist.append(w_dist)

                # Train G
                for p in self.D.parameters():
                    p.requires_grad = False  # to avoid computation

                g_err = self.generator_train()
                G_losses.append(g_err)
            if pos_seqs:
                # after epoch training remove the fake data
                self.remove_old_indices(len(pos_seqs))
            summary_string = 'Epoch{0}/{1}: d_real_loss:{2:.2f},d_fake_loss:{3:.2f},d_total_loss:{4:.2f},G_total_loss:{5:.2f},W_dist:{6:.2f}' \
                .format(epoch, self.n_epochs, np.mean(d_real_losses), np.mean(d_fake_losses), np.mean(D_losses),
                        np.mean(G_losses), np.mean(W_dist))

            self.logger.info(summary_string)
            loss_f.write(str(np.mean(d_real_losses))+','+str(np.mean(d_fake_losses))+','+str(np.mean(D_losses))+
                         ','+str(np.mean(G_losses))+','+str(np.mean(W_dist))+'\n')

        if old_epoch < self.n_epochs:
            self.save_model(self.n_epochs)
        if self.n_sequences > 0:
            self.gan_sequences()
        loss_f.close()
        return None
