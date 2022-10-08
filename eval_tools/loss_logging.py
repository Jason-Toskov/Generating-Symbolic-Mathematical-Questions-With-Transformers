import matplotlib.pyplot as plt

class LossLogger():
    def __init__(self, params, loss_to_track):
        # TODO: change this to allow for an abitrary number of losses
        self.params = params

        self.loss_lists = {
            'epoch': {},
            'batch': {}
        }

        self.current_batch_loss = {}

        self.loss_to_track = loss_to_track

        self.tracked_loss = 10^5


    def add_loss(self, loss_dict):
        for loss_type, loss_val in loss_dict.items():
            if loss_type not in self.current_batch_loss:
                self.current_batch_loss[loss_type] = []
            self.current_batch_loss[loss_type].append(loss_val)

    def end_epoch(self):
        for loss_type, loss_val in self.current_batch_loss.items():
            if loss_type not in self.loss_lists['epoch']:
                self.loss_lists['epoch'][loss_type] = []
                self.loss_lists['batch'][loss_type] = []

            self.loss_lists['batch'][loss_type] += loss_val
            mean_loss = self.mean(loss_val)
            self.loss_lists['epoch'][loss_type].append(mean_loss)

            print('Average %s loss: %.4f' % (loss_type, mean_loss))
        
        self.tracked_loss = 0
        for loss_type, weight in self.loss_to_track.items():
            self.tracked_loss += self.loss_lists['epoch'][loss_type][-1]*weight

        self.plot_losses()
        self.current_batch_loss = {}

    def plot_losses(self):
        for time_metric, loss_dict in self.loss_lists.items():
            for loss_type, loss_val in loss_dict.items():
                plt.plot(loss_val, label=loss_type)
            plt.xlabel(time_metric)
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            plt.savefig(self.params.dump_path + 'per_' + time_metric + '_loss.png')
            plt.close()

    def get_description(self):
        desc = ''
        for loss_type, loss_val in self.current_batch_loss.items():
            desc += '%s: %.2f, ' % (loss_type,loss_val[-1])
        return desc

    def mean(self, l):
        return sum(l)/len(l)

