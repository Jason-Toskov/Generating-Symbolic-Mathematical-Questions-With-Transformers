import torch
import torch.nn.functional as F
from utils import ModelType
from eval_tools.loss_logging import LossLogger
from eq_parse_helper import get_math_functions_used
from utils import to_cuda, gumbel_softmax

class Trainer():
    def __init__(self, models, env, params):
        """
        Initialize trainer.
        """
        # modules / params
        self.models = models
        self.params = params
        self.env = env

        self.epoch_size = params.epoch_size

        # set parameters
        self.set_parameters()

        self.init_optimizers() 

        # Parse data path
        s = [x.split(',') for x in params.reload_data.split(';') if len(x) > 0]
        self.data_path = {task: (train_path, valid_path, test_path) for task, train_path, valid_path, test_path in s}

        # this both creates the dataset and the dataloader
        self.ds_dict = {}
        self.dl_dict = {}
        for task in params.tasks:
            ds, dl = env.create_train_iterator(task, params, self.data_path)
            self.ds_dict[task] = ds
            self.dl_dict[task] = iter(dl)

        #self.train_datapoints = [prob[0] for dset in ds_list for prob in dset.data]

        self.n_equations = 0
        self.epoch = 0

        self.best_loss = 10^5

        # Can add multiple losses here, {<type>:<weight>}
        if self.params.model_type == ModelType.BASIC:
            self.loss_to_track = {'MLE':1}
        elif self.params.model_type == ModelType.TransGAN:
            self.loss_to_track = {'MLE':1}

        self.loss_logger = LossLogger(self.params, self.loss_to_track)

        # breakpoint()

    def find_eq_in_dataset(self, eq):
        eq_test = ' '.join(eq)
        for k, ds in self.ds_dict.items():
            for prob in ds.data:
                if eq_test == prob[0]:
                    return True
        
        return False

    def init_optimizers(self):
        if self.params.model_type == ModelType.BASIC:
            self.optimizer = torch.optim.Adam(self.parameters['gen'], lr=self.params.learning_rate)
        elif self.params.model_type == ModelType.TransGAN:
            self.optimizer = {
                'gen_opt': torch.optim.Adam(self.parameters['gen'], lr=self.params.learning_rate),
                'disc_opt':torch.optim.Adam(self.parameters['disc'], lr=self.params.learning_rate)
            }


    def set_parameters(self):
        """
        Set parameters.
        """
        self.parameters = {}

        named_params = []
        for key in ['gen_encoder', 'gen_decoder']:
            named_params.extend([(k, p) for k, p in self.models[key].named_parameters() if p.requires_grad])
        self.parameters['gen'] = [p for k, p in named_params]

        if self.params.model_type == ModelType.TransGAN:
            param_temp = [(k, p) for k, p in self.models['disc'].named_parameters() if p.requires_grad]
            self.parameters['disc'] = [p for k, p in param_temp]

        total_params = 0
        for v in self.parameters.values():
            for p in v:
                total_params += len(p)
        print("Found %i parameters." % (total_params))

    def end_epoch(self):
        self.epoch += 1
        self.loss_logger.end_epoch()
        if self.loss_logger.tracked_loss < self.best_loss:
            self.best_loss = self.loss_logger.tracked_loss
            print('New lowest loss: %.4f' % (self.best_loss))
            self.save_model()
            print('Model saved!')

    def get_batch(self, task):
        batch = next(self.dl_dict[task])
        return batch

    def train_step(self, task):
        if self.params.model_type == ModelType.BASIC:
            self.enc_dec_step_basic(task)
        elif self.params.model_type == ModelType.TransGAN:
            self.enc_dec_step_GAN(task)

    def save_model(self):
        if self.params.model_type == ModelType.BASIC:

            data = {
                'epoch': self.epoch,
                # 'env': self.env,
                # 'params': self.params,
                'gen_enc_weights': self.models["gen_encoder"].state_dict(),
                'gen_dec_weights': self.models["gen_decoder"].state_dict()
            }

            torch.save(data, self.params.dump_path+'basic_model_best.pt')
        elif self.params.model_type == ModelType.TransGAN:
            data = {
                'epoch': self.epoch,
                # 'env': self.env,
                # 'params': self.params,
                'gen_enc_weights': self.models["gen_encoder"].state_dict(),
                'gen_dec_weights': self.models["gen_decoder"].state_dict(),
                'disc_weights': self.models["disc"].state_dict()
            }

            torch.save(data, self.params.dump_path+'transgan_model_best.pt')

    def fwd_gen(self, encoder, decoder, eq_start, eq_start_len, eq_out, eq_out_len):
        # target words to predict
        alen = torch.arange(eq_out_len.max(), dtype=torch.long, device=eq_out_len.device) # -> max seq len in batch 
        # do not predict anything given the last target word
        pred_mask = alen[:, None] < eq_out_len[None] - 1 # -> Mask for sequences (True when not EOS)
        y = eq_out[1:].masked_select(pred_mask[:-1]) # -> y is a flat tensor of all the tokens to be predicted
        assert len(y) == (eq_out_len - 1).sum().item()

        # cuda
        eq_start, eq_start_len, eq_out, eq_out_len, y = to_cuda(eq_start, eq_start_len, eq_out, eq_out_len, y)

        # forward pass
        encoded = encoder('fwd', x=eq_start, lengths=eq_start_len, causal=False) # -> shape: (input_size, bs, model_output_dim)
        # -> encoded shape: (input_size, bs, model_output_dim)

        decoded = decoder('fwd', x=eq_out, lengths=eq_out_len, causal=True, src_enc=encoded.transpose(0, 1), src_len=eq_start_len)
        # -> decoded shape: (output_size, bs, model_output_dim)

        return encoded, decoded, pred_mask, y

    def enc_dec_step_basic(self, task):
        """
        Encoding / decoding step.
        """
        encoder = self.models["gen_encoder"]
        decoder = self.models["gen_decoder"]

        encoder.train()
        decoder.train()

        # unpack batch
        # x1 = equation to solve, x2 = equation solution
        # Hence, we only care about x1 for equation generation
        # TODO: will want to stop x2 from loading at all to save memory
        batch = self.get_batch(task)
        (x1, len1), (x2, len2), (func, func_len), (step_, step_len), (cond, cond_len), _ = batch
        # breakpoint()

        eq_start = cond
        eq_start_len = cond_len

        # Output should be the whole equation
        eq_out = x1
        eq_out_len = len1

        encoded, decoded, pred_mask, y = self.fwd_gen(encoder, decoder, eq_start, eq_start_len, eq_out, eq_out_len) 

        # breakpoint() 

        # Prediction step to get loss
        _, loss = decoder('predict', tensor=decoded, pred_mask=pred_mask, y=y, get_scores=False)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.loss_logger.add_loss({'MLE':loss.item()})

        self.n_equations += self.params.batch_size

    def enc_dec_step_GAN(self, task):

        real_label = 1.
        fake_label = 0.

        criterion = torch.nn.BCEWithLogitsLoss()

        gen_encoder = self.models["gen_encoder"]
        gen_decoder = self.models["gen_decoder"]
        disc = self.models["disc"]

        #####################################
        #        Train discriminator        #
        #####################################
        
        disc.zero_grad()
        self.optimizer['disc_opt'].zero_grad()

        #### Train on real batch:
    
        # get batch of real data
        batch = self.get_batch(task)
        (x1, len1), (x2, len2), (func, func_len), (step_, step_len), (cond, cond_len), _ = batch

        label_real = torch.full((self.params.batch_size,), real_label, dtype=torch.float)
        x1, len1, label_real = to_cuda(x1, len1, label_real)

        # Fwd pass real batch through D
        # real_input = F.one_hot(x1, num_classes=self.params.n_words).type(torch.float32)
        prediction_disc_real = disc(x1, len1).squeeze()

        # Caclulate loss on real batch
        errD_real = criterion(prediction_disc_real, label_real)
        errD_real.backward()

        #### Train on fake batch:
        batch = self.get_batch(task)
        (x1, len1), (x2, len2), (func, func_len), (step_, step_len), (cond, cond_len), _ = batch
        x1, len1 = to_cuda(x1, len1)
        # Generate fake sequences
        eq_start = cond
        eq_start_len = cond_len

        eq_out = x1
        eq_out_len = len1

        # TODO: right now this is the standard 'teacher forcing'-esque training
        # Should this be generating tokens one-by-one based on its own output instead?
        encoded, decoded, pred_mask, y = self.fwd_gen(gen_encoder, gen_decoder, eq_start, eq_start_len, eq_out, eq_out_len)

        projected_dec = gen_decoder.proj(decoded)
        gumbel_out = gumbel_softmax(projected_dec, 0.5)

        # Refresh label
        label_fake = to_cuda(torch.full((self.params.batch_size,), fake_label, dtype=torch.float))[0]

        # Detach grad and then pass through D
        # First add zeros to the start of the seq to match real
        ones_vec = to_cuda(torch.zeros(gumbel_out.shape[-1], dtype=torch.float32))[0]
        ones_vec[1] = 1.
        gumbel_out[~pred_mask] = ones_vec
        zeros_vec = to_cuda(torch.zeros((1,*gumbel_out.shape[1:])))[0]
        fake_gen = torch.cat((zeros_vec, gumbel_out[:-1]))

        # This takes the argmax to train the discriminator
        fake_dicretized = fake_gen.argmax(dim=2)

        prediction_disc_fake = disc(fake_dicretized.detach(), eq_out_len).squeeze()

        # Calculate loss and .backward()
        errD_fake = criterion(prediction_disc_fake, label_fake)
        errD_fake.backward()

        #### Update D
        # Sum errors
        errD = errD_real + errD_fake

        # Step optimizer
        self.optimizer['disc_opt'].step()

        #####################################
        #          Train generator          #
        #####################################

        gen_encoder.zero_grad()
        gen_decoder.zero_grad()
        self.optimizer['gen_opt'].zero_grad()

        # get batch
        batch = self.get_batch(task)
        (x1, len1), (x2, len2), (func, func_len), (step_, step_len), (cond, cond_len), _ = batch

        x1, len1 = to_cuda(x1, len1)

        eq_start = cond
        eq_start_len = cond_len

        eq_out = x1
        eq_out_len = len1

        encoded, decoded, pred_mask, y = self.fwd_gen(gen_encoder, gen_decoder, eq_start, eq_start_len, eq_out, eq_out_len)

        projected_dec = gen_decoder.proj(decoded)
        gumbel_out = gumbel_softmax(projected_dec, 0.5)

        ones_vec = to_cuda(torch.zeros(gumbel_out.shape[-1], dtype=torch.float32))[0]
        ones_vec[1] = 1.
        gumbel_out[~pred_mask] = ones_vec
        zeros_vec = to_cuda(torch.zeros((1,*gumbel_out.shape[1:])))[0]
        fake_gen = torch.cat((zeros_vec, gumbel_out[:-1]))

        # For the generator we want fake input to get a 'real' label
        label_real2 = to_cuda(torch.full((self.params.batch_size,), real_label, dtype=torch.float))[0]

        # Prediction step to get loss
        _, mle_loss = gen_decoder('predict', tensor=decoded, pred_mask=pred_mask, y=y, get_scores=False)
        # mle_loss.backward()

        # Pass fake sequences through generator now
        prediction_gen = disc(fake_gen, eq_out_len).squeeze()

        # Calculate loss and .backward() and step optimizer
        errG = criterion(prediction_gen, label_real2)
        # errG.backward()

        gen_err = mle_loss #+ 0.5*errG
        gen_err.backward()
        self.optimizer['gen_opt'].step()

        loss_dict = {
            'G': errG.item(),
            'D [real]': errD_real.item(),
            'D [fake]': errD_fake.item(),
            'MLE': mle_loss.item()
        }

        self.loss_logger.add_loss(loss_dict)
        # if self.epoch > 5:
        #     breakpoint()

        self.n_equations += self.params.batch_size
