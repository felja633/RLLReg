import torch
import torch.autograd
from torch.utils.data import DataLoader
from evaluation.evaluate import benchmark
from pathlib import Path
import glob

class trainer:
    def __init__(self, num_epochs, actor, optimizer, scheduler, job_name, evaluator=None,
                 collate_fn=None, update_dataset=True):
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.actor = actor
        self.evaluator = evaluator
        self.job_name = job_name
        self.collate_fn = collate_fn
        self.update_dataset = update_dataset

    def train(self, dataset, batch_size=1, valdata=None, num_workers=0, load_prev=False, epoch=None):

        if load_prev:
            current_epoch = self.load_checkpoint(epoch=epoch) + 1
        else:
            current_epoch = 0

        for epoch in range(current_epoch, self.num_epochs):
            self.actor.train()
            loss_acc = 0
            iter = 0
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=self.collate_fn)
            self.optimizer.zero_grad()
            val=0
            for batch in loader:
                loss, val = self.actor(batch, epoch)
                self.optimizer.zero_grad()
                print_str = '[Epoch %d, batch %d of %d] ' % (epoch, iter, loader.__len__() - 1)
                # only backpropagate of loss is non-zero
                if loss > 0.0:
                    iter+=1
                    loss_acc += loss.item()
                    print(print_str, "loss:", loss.item(), "loss_acc: ", loss_acc/iter)
                    loss.backward()
                    self.optimizer.step()

            self.scheduler.step(epoch=epoch)

            self.save_checkpoint(self.job_name, epoch, val)

            if not valdata is None:
                self.validate(self.job_name, valdata, batch_size, epoch)

            if self.update_dataset:
                dataset.generate_samples()

    def validate(self, job_name, valdata, batch_size, epoch):
        val_model = type(self.actor.model)(self.actor.model.params)  # get a new instance
        val_model.load_state_dict(self.actor.model.state_dict())  # copy weights and stuff
        val_model.train(mode=False)
        eval_info = benchmark([val_model], valdata, self.job_name, batch_size=batch_size, plot=False,
                              epoch=epoch, collate_fn=self.collate_fn)

        path = job_name + "/checkpoints"
        Path(path).mkdir(parents=True, exist_ok=True)
        txtfile = path + "/val" + "_log.txt"
        f = open(txtfile, "a+")
        f.write("Epoch: %s R recall 4 deg: %s t recall 0.3 meters: %s total: %s \n" % (str(epoch),
                                                     eval_info[val_model.params.name]["R_recall"],
                                                     eval_info[val_model.params.name]["t_recall"],
                                                     eval_info[val_model.params.name]["total_recall"]))
        f.close()

    def save_checkpoint(self, job_name, epoch, val):
        path = job_name + "/checkpoints"
        Path(path).mkdir(parents=True, exist_ok=True)
        filename = path + "/checkpoint" + str(epoch) + ".pth"
        txtfile = path + "/checkpoint" + "_log.txt"
        f = open(txtfile, "a+")
        f.write("Epoch: %s num_valid: %s \n" % (str(epoch), str(val)))
        f.close()

        if epoch % 5 == 0:
            actor_type = type(self.actor).__name__
            net_type = type(self.actor.model).__name__
            state = {
                'epoch': epoch,
                'actor_type': actor_type,
                'net_type': net_type,
                'actor': self.actor.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            torch.save(state, filename)

    def load_checkpoint(self, epoch=None):
        path = self.job_name + "/checkpoints"
        if not epoch is None:
            filename = path + "/checkpoint" + str(epoch) + ".pth"
            state = torch.load(filename)
            self.actor.load_state_dict(state["actor"])
            epoch = state["epoch"]
            self.optimizer.load_state_dict(state["optimizer"])
        else:
            # load latest
            checkpoint_list = sorted(glob.glob('{}/{}/checkpoint*.pth'.format(self.job_name,
                                                                              "checkpoints")))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return

            state = torch.load(checkpoint_path)
            self.actor.load_state_dict(state["actor"])
            epoch = state["epoch"]
            self.optimizer.load_state_dict(state["optimizer"])

        return epoch
