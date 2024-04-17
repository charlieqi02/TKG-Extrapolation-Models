"""Train Temporal Knowledge Graph embeddings for extrapolation."""
import argparse
import json
import logging
import os

import torch
import torch.optim

import models
import optimizers.regularizers as regularizers
from datasets.auxdata import load_auxdata
from datasets.tkg_dataset import TKGDataset
from models import all_models
from optimizers.tkg_optimizer import TKGOptimizer
from recorder import Recorder
from utils.train import count_params, get_savedir, set_seed

parser = argparse.ArgumentParser(
    description="Temporal Knowledge Graph Extrapolation"
)
# General arguments
parser.add_argument(
    "--des", default="", type=str, help="Description of the run, will be saved in config.json")
parser.add_argument(
    "--dataset", default="ICEWS14s", choices=["ICEWS05-15", "ICEWS18", "ICEWS14s", "GDELT", "WIKI", "YAGO"],
    help="Temporal Knowledge Graph dataset")
parser.add_argument(
    "--rank", default=200, type=int, help="Embedding dimension")
parser.add_argument(
    "--seq_len", default=3, type=int, help="Nearest snaps used as history")
parser.add_argument(
    "--model", choices=all_models, default="SimREGCN", help="Temporal Knowledge Graph embedding model")
parser.add_argument(
    "--regularizer", choices=["R0", "Static"], default="R0", help="Regularizer")
parser.add_argument(
    "--optimizer", choices=["Adagrad", "Adam", "SparseAdam"], default="Adam", help="Optimizer")
parser.add_argument(
    "--max_epochs", default=50, type=int, help="Maximum number of epochs to train for")
parser.add_argument(
    "--reg_alpha", default=0.5, type=float, help="Weight of regularization in loss")
parser.add_argument(
    "--learning_rate", default=1e-1, type=float, help="Learning rate")
parser.add_argument(
    "--label_smoothing", default=0.2, type=float, help="Label smoothing used in cross entrophy loss")
parser.add_argument(
    "--dropout", default=0, type=float, help="Dropout rate")
parser.add_argument(
    "--batch_size", default=1, type=int, help="Batch size")
parser.add_argument(
    "--continue_train", default=False, action='store_true', help="If use checkpoint to continue training")
parser.add_argument(
    "--continue_dir", default="", type=str, help="Dir to use checkpoint to continue training")
parser.add_argument(
    "--seed", default=3407, type=int, help="Random seed")
parser.add_argument(
    "--gpu", type=int, default=-1, help="gpu")
parser.add_argument(
    "--grad_norm", type=float, default=1.0, help="norm to clip gradient to")
parser.add_argument(
    "--debug", default=False, action='store_true', help="Decrease data length for debugging")
parser.add_argument(
    "--debug_len", default=10, type=int, help="Decreased data length for debugging")


## Advanced record option
parser.add_argument(
    "--record_atth", default=False, action='store_true', help="Entity embedding l2 norm and multi-c record option")


# model specific arguments
## RE-GCN
parser.add_argument(
    "--use_static", default=False, action='store_true', help="Using static graph regularition (contraint) if True")
parser.add_argument(
    "--discount", default=False, action='store_true', help="Ascending pace of the angle {angle, angle*2} if True")
parser.add_argument(
    "--angle", default=10, type=float, help="Angle contraining difference between static emb and ent emb")
parser.add_argument(
    "--num_rgcn_layers", default=1, type=int, help="Number of RGCN layers in each evulotion cell")
parser.add_argument(
    "--self_loop", default=False, action='store_true', help="Using self-loop for each node in RGCN if True")
parser.add_argument(
    "--channel", default=50, type=int, help="Channels in ConvTransE")
parser.add_argument(
    "--kernel_size", default=3, type=int, help="Kernel size in ConvTransE")


## ATTH
parser.add_argument(
    "--init_size", default=1e-3, type=float, help="Initial embeddings' scale")
parser.add_argument(
    "--multi_c", action="store_true", help="Multiple curvatures per relation")
parser.add_argument(
    "--dtype", default="double", type=str, choices=["single", "double"], help="Machine precision")



def train(args):
    set_seed(args.seed)
    torch.cuda.set_device(args.gpu)
    save_dir = get_savedir(args) 
    
    # file logger
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s \t %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=os.path.join(save_dir, "train.log"))
    # stdout logger
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s \t %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    logging.info("Saving logs in: {}".format(save_dir))
    # save config
    with open(os.path.join(save_dir, "config.json"), "w") as fjson:
        json.dump(vars(args), fjson, indent=2)
    logging.info("Config saved as config.json")
        

    # create dataset
    dataset = TKGDataset(args.dataset, args.debug, args.debug_len)
    args.sizes = dataset.get_shape()    
    # load data
    train_list = dataset.get_snaps("train")
    valid_list, valid_ans4tf = dataset.get_snaps("valid"), dataset.get_ans4tf('valid')
    test_list , test_ans4tf  = dataset.get_snaps("test"),  dataset.get_ans4tf('test')
    # load auxiliary data
    aux_data, args = load_auxdata(args)


    # create model
    model = getattr(models, args.model)(args).to(args.gpu)
    total = count_params(model)
    logging.info("Total number of parameters {}".format(total))
    # get optimize things
    regularizer = getattr(regularizers, args.regularizer)(args)
    optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
    # get recorder (include load checkpoint)
    recorder = Recorder(save_dir, args, model, optim_method, 
                        aux_data, train_list, valid_list, test_list)
    # get optimizer
    optimizer = TKGOptimizer(model, regularizer, optim_method, args.reg_alpha, args.label_smoothing,
                            args.seq_len, args.grad_norm)  


    logging.info("Start training")
    for step in range(recorder.start_epoch, args.max_epochs):
        # Train step
        model.train()
        train_records = optimizer.epoch(train_list, aux_data)
        recorder.train_recording(train_records, step)

        # Valid step
        model.eval()
        valid_records = optimizer.get_valid_LossAndMetric(train_list, valid_list, aux_data, valid_ans4tf)
        recorder.valid_recording(valid_records, step, 'valid', model)

        # Test step (Shouldn't do, but ...)
        test_records = optimizer.get_valid_LossAndMetric(train_list+valid_list, test_list, aux_data, test_ans4tf)
        recorder.valid_recording(test_records, step, 'test', model)

        # Inner record
        recorder.inner_recording(model, step)

        # save checkpoint and best models (for test/valid raw/filter each)
        recorder.save_checkpoint(model, optim_method, step)
    recorder.close_writer()
    recorder.print_best_models()


if __name__ == "__main__":
    try:
        train(parser.parse_args())
    except Exception as e:
        logging.error(e, exc_info=True)
        
        
        
        