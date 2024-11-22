import argparse
import os
import datetime

# ---- Usefull Utilities ----
def mkdir(path):
    '''create a single empty directory if it didn't exist
    Parameters: path (str) -- a single directory path'''
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    '''create empty directories if they don't exist
    Parameters: paths (str list) -- a list of directory paths'''
    rmdirs(paths)
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def rmdirs(paths):
    if os.path.exists(paths):
        for file in os.listdir(paths): 
            file_path = os.path.join(paths, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                rmdirs(file_path)
        os.rmdir(paths)


class Options():
    ''' This class defines argsions
        adapted from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/options
    '''
    def __init__(self, accelerator = None):
        self.initialized = False
        self.accelerator = accelerator

    def initialize(self, parser):
        ''' set up arguments '''

        parser.add_argument('--experiment_dir', type=str, default='experiment')
        parser.add_argument('--message_file', type=str, default='messages.csv')
        parser.add_argument('--datadir', type = str, default='natural-instructions', help='data')
        parser.add_argument('--model_name', type = str, default='gpt2')
        parser.add_argument('--local_model_path', type = str, default=None)
        parser.add_argument('--device', type = str, default='cuda:0')
        parser.add_argument('--tasks', type = str, default='dbpedia,amazon,yahoo,agnews')

        parser.add_argument('--seed', type = int, default=32)
        parser.add_argument('--train_epochs', type = int, default=10)
        parser.add_argument('--warm_up_epochs', type = int, default=3)
        parser.add_argument('--scheduler_type', type = str, default="linear")

        parser.add_argument('--bs', type = int, default=16, help='batch size')
        parser.add_argument('--lr', type = float, default=2e-5, help='learning rate')
        parser.add_argument('--wd', type = float, default=0.01, help='weight decay')
        parser.add_argument('--save_model', default=False, action='store_true')
        parser.add_argument('--print_generations', default=False, action='store_true')

        parser.add_argument('--use_lora', default=False, action='store_true')
        parser.add_argument('--lora_r', type = int, default=16)
        parser.add_argument('--lora_alpha', type = int, default=16)
        parser.add_argument('--lora_dropout', type = float, default=0.1)
        parser.add_argument('--lora_target_modules', type = str, default=None)
        parser.add_argument('--lora_fan_in_fan_out', default=False, action='store_true')
        parser.add_argument('--llambda', type = float, default=0.5)
        parser.add_argument('--normalized_lora', default=False, action='store_true')
        
        parser.add_argument('--lora_target_module_layers_idx', type = str, default=None)
        parser.add_argument('--lora_target_module_proj_type', type = str, default=None)


        # parser.add_argument('--cosine_annealing',default=False, action='store_true', help='whether to use cosine annealing or not')
        # parser.add_argument('--grad_clipping', type = float, default=None, help='max grad norm for clipping')
        self.initialized = True
        return parser

    def get_argsions(self):
        ''' get argsions from parser '''
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser()
            parser = self.initialize(parser)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_argsions(self, args):
        ''' Print and save argsions
            It will print both current argsions and default values(if different).
            It will save argsions into a text file / [checkpoints_dir] / args.txt
        ''' 
        message = str(datetime.datetime.now())
        message += '\n----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        if self.accelerator and self.accelerator.is_main_process or not self.accelerator:
            expr_dir = args.experiment_dir
            mkdirs(expr_dir) # first remove existing directory and then create a new one
            file_name = os.path.join(expr_dir, 'args.txt')
            with open(file_name, 'wt') as args_file:
                args_file.write(message)
                args_file.write('\n')
    
    def parse(self):
        ''' Parse our argsions, create checkpoints directory suffix, and set up gpu device. '''
        args = self.get_argsions()
        
        # converting task_ids to a list object
        args.tasks = args.tasks.split(',')
        
        if args.lora_target_modules:
            args.lora_target_modules = args.lora_target_modules.split(',')
        if args.lora_target_module_layers_idx:
            args.lora_target_module_layers_idx = args.lora_target_module_layers_idx.split(',')
        if args.lora_target_module_proj_type:
            args.lora_target_module_proj_type = args.lora_target_module_proj_type.split(',')

        self.print_argsions(args)
        self.args = args
        return self.args