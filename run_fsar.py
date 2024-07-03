import torch
import numpy as np
import argparse
import os
from utils.util import print_and_log, get_log_files, TestAccuracies, loss, aggregate_accuracy
from models.model import CNN_GCN_FSAR
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet TensorFlow warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
import video_reader
from utils.config import Config
import models.utils.optimizer as optim 
import scipy.io
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
def main():
    learner = Learner()
    #learner.run()
    learner.test()
    

class Learner:
    def __init__(self):
        self.cfg = Config(load=True)
        self.args = self.parse_command_line()

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir, self.args.resume_from_checkpoint, False)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        
        gpu_device = 'cuda'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        torch.cuda.empty_cache()
        self.model = self.init_model()
        self.train_set, self.validation_set, self.test_set = self.init_data()

        self.vd = video_reader.VideoDataset(self.args)
        self.video_loader = torch.utils.data.DataLoader(self.vd, batch_size=1, num_workers=self.args.num_workers)
        
        self.loss = loss
        self.accuracy_fn = aggregate_accuracy
        
        if self.args.opt == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.opt == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        #  Construct the optimizer.
        self.optimizer = optim.construct_optimizer(self.model, self.cfg)
            
        self.test_accuracies = TestAccuracies(self.test_set)
        
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.args.sch, gamma=0.1)
        
        self.start_iteration = 0
        if self.args.resume_from_checkpoint:
            self.load_checkpoint()
        self.optimizer.zero_grad()

    def init_model(self):
        model = CNN_GCN_FSAR(self.args) # CNN_TRX(self.args)#
        model = model.to(self.device)
        if self.args.num_gpus > 1:
            model.distribute_model()
        return model

    def init_data(self):
        train_set = [self.args.dataset]
        validation_set = [self.args.dataset]
        test_set = [self.args.dataset]
        return train_set, validation_set, test_set


    """
    Command line parser
    """
    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--dataset", choices=["ssv2", "kinetics", "hmdb", "ucf","ucf_crime"], default="hmdb", help="Dataset to use.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=0.00005, help="Learning rate.")
        parser.add_argument("--tasks_per_batch", type=int, default=16, help="Number of tasks between parameter optimizations.")
        parser.add_argument("--checkpoint_dir", "-c", default='checkpoint', help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_path", "-m", default=None, help="Path to model to load and test.")
        parser.add_argument("--resume_from_checkpoint", "-r", dest="resume_from_checkpoint", default=False, action="store_true", help="Restart from latest checkpoint.")
        parser.add_argument("--way", type=int, default=5, help="Way of each task.")
        parser.add_argument("--shot", type=int, default=1, help="Shots per class.")
        parser.add_argument("--query_per_class", type=int, default=5, help="Target samples (i.e. queries) per class used for training.")
        parser.add_argument("--query_per_class_test", type=int, default=1, help="Target samples (i.e. queries) per class used for testing.")
        parser.add_argument('--test_iters', nargs='+', type=int,default=[5000000,800000],help='iterations to test at. Default is for ssv2 otam split.')
        parser.add_argument("--num_test_tasks", type=int, default=1, help="number of random tasks to test on.")
        parser.add_argument("--print_freq", type=int, default=10, help="print and log every n iterations.")
        parser.add_argument("--seq_len", type=int, default=8, help="Frames per video.")
        parser.add_argument("--num_workers", type=int, default=10, help="Num dataloader workers.")
        parser.add_argument("--opt", choices=["adam", "sgd"], default="adam", help="Optimizer")
        parser.add_argument("--trans_dropout", type=int, default=0.1, help="Transformer dropout")
        parser.add_argument("--save_freq", type=int, default=1000, help="Number of iterations between checkpoint saves.")
        parser.add_argument("--img_size", type=int, default=224, help="Input image size to the CNN after cropping.")
        parser.add_argument("--scratch", choices=["bc", "bp"], default="bp", help="directory containing dataset, splits, and checkpoint saves.")
        parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to split the ResNet over")
        parser.add_argument("--debug_loader", default=False, action="store_true", help="Load 1 vid per class for debugging")
        parser.add_argument("--split", type=int, default=3, help="Dataset split.")
        parser.add_argument('--sch', nargs='+', type=int, help='iters to drop learning rate', default=[1000000])
        args = parser.parse_args()
        
        if args.dataset == "ssv2":
            args.traintestlist = "/media/psdz/新加卷/zjm/TgGTM/splits/ssv2_OTAM"
            args.path = "/media/psdz/新加卷/zjm/ssv2/ssv2_frames"
        elif args.dataset == "kinetics":
            args.traintestlist = "/media/psdz/新加卷/zjm/TgGTM/splits/kinetics"
            args.path = "/media/psdz/新加卷/zjm/PyVideoFramesExtractor/kinetics_frames"
        elif args.dataset == "ucf":
            args.traintestlist = "/media/psdz/新加卷/zjm/TgGTM/splits/ucf_ARN"
            args.path = "/media/psdz/新加卷/zjm/PyVideoFramesExtractor/frame"
        elif args.dataset == "hmdb":
            args.traintestlist = "/media/psdz/新加卷/zjm/TgGTM/splits/hmdb_ARN"
            args.path = "/media/psdz/新加卷/zjm/hmdb51_frames"

        return args

    def run(self):
                train_accuracies = []
                losses = []
                total_iterations = self.cfg.TRAIN.NUM_TRAIN_TASKS

                iteration = self.start_iteration
                for task_dict in self.video_loader:
                    if iteration >= total_iterations:
                        break
                    iteration += 1
                    torch.set_grad_enabled(True)

                    task_loss, task_accuracy = self.train_task(task_dict)
                    train_accuracies.append(task_accuracy.item())
                    losses.append(task_loss.item())
                    
                    cur_epoch = iteration//self.cfg.SOLVER.STEPS_ITER
                    data_size = self.cfg.SOLVER.STEPS_ITER
                   
                    lr = optim.get_epoch_lr(cur_epoch + self.cfg.TRAIN.NUM_FOLDS * float(iteration) / data_size, self.cfg)
                    optim.set_lr(self.optimizer, lr)
                    # print(lr)
                    # optimize
                    if ((iteration + 1) % self.args.tasks_per_batch == 0) or (iteration == (total_iterations - 1)):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    self.scheduler.step()
                    if (iteration + 1) % self.args.print_freq == 0:
                        # print training stats
                        print_and_log(self.logfile,'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'
                                      .format(iteration + 1, total_iterations, np.array(losses).mean(),
                                              np.array(train_accuracies).mean()))
                        train_accuracies = []
                        losses = []

                    if ((iteration + 1) % self.args.save_freq == 0) and (iteration + 1) != total_iterations:
                        self.save_checkpoint(iteration + 1)


                    if ((iteration + 1) in self.args.test_iters) and (iteration + 1) != total_iterations:
                        accuracy_dict = self.test()
                        print(accuracy_dict)
                        # self.test_accuracies.printf(self.logfile, accuracy_dict)
                # save the final model
                torch.save(self.model.state_dict(), self.checkpoint_path_final)
                self.test()
                self.logfile.close()

    def train_task(self, task_dict):
        
        model_dict = self.model(task_dict)
        target_logits = F.cross_entropy(model_dict['logits'],  task_dict["target_labels"][0].long().to(self.device))
       
        class_logits = F.cross_entropy(model_dict["class_text_logits"], task_dict["real_target_labels"][0].long().to(self.device))
        task_loss = ( target_logits +class_logits) / self.args.tasks_per_batch

        task_accuracy = self.accuracy_fn(model_dict['logits'].unsqueeze(0), task_dict["target_labels"][0].long().to(self.device))
   

        task_loss.backward(retain_graph=False)
        return task_loss, task_accuracy

    def test(self):
        class_correct = np.zeros(101, dtype=float)
        class_total = np.zeros(101, dtype=float)
        self.model.eval()
        with torch.no_grad():
            self.video_loader.dataset.train = False
            accuracy_dict = {}
            accuracies = []
            iteration = 0
            item = self.args.dataset
            with tqdm(total=self.args.num_test_tasks) as bar:
                for task_dict in self.video_loader:
                    if iteration >= self.args.num_test_tasks:
                        break
                    iteration += 1
                    bar.set_description('testing:')
                    bar.update(1)

                    model_dict = self.model(task_dict)
                    target_logits = model_dict['logits']
                    
                    accuracy = self.accuracy_fn(target_logits.unsqueeze(0), task_dict["target_labels"][0].long().to(target_logits.device))
                    accuracies.append(accuracy.item())
                    del target_logits
                    real_label = task_dict['real_target_labels'][0].numpy().astype('int32')
                    correct = torch.eq( torch.argmax(model_dict['logits'], dim=-1), task_dict["target_labels"][0].long().to(self.device))
                    for i,l in enumerate(correct):
                        if l :
                            class_correct[real_label[i]] +=1
                    class_total[real_label]+=1
                accuracy = np.array(accuracies).mean() * 100.0
                confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

                accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence}
                print(accuracy_dict)
                scipy.io.savemat('acc_test_{}shot_{}.mat'.format(self.args.shot,self.args.dataset),
                     {'class_total': class_total,'class_correct':class_correct})
                self.video_loader.dataset.train = True
        self.model.train()
        
        return accuracy_dict

    def prepare_task(self, task_dict, images_to_device = True):
        context_images, context_labels = task_dict['support_set'][0], task_dict['support_labels'][0]
        target_images, target_labels = task_dict['target_set'][0], task_dict['target_labels'][0]
        # batch_class_list = task_dict['batch_class_list'][0]

        if images_to_device:
            context_images = context_images.to(self.device)
            target_images = target_images.to(self.device)
        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)

        return context_images, target_images, context_labels, target_labels  

    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]


    def save_checkpoint(self, iteration):
        d = {'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint{}.pt'.format(iteration)))
        torch.save(d, os.path.join(self.checkpoint_dir, 'checkpoint.pt'))

    def load_checkpoint(self):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'HMDB51-1SHOT-81.6.pt'))
        # self.start_iteration = checkpoint['iteration']
        # self.model.load_state_dict(checkpoint['model_state_dict'])
        # s = self.model.state_dict()
        self.model.load_state_dict(checkpoint)
        print('loading checkpoint!')
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.scheduler.load_state_dict(checkpoint['scheduler'])


if __name__ == "__main__":
    main()
