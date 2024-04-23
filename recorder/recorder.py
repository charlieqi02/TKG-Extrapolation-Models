from .recorder_bm import RecoderBestModel
from .recorder_cp import RecoderCheckpoint
from .recorder_lm import RecorderLossAndMetrics
from .recorder_nn import RecorderInnerModel


class Recorder:
    """For parsing Loss/Metric logging, TensorBoard writing, Model saving and loading"""
    def __init__(self, save_dir, args, model, optimizer,
                         aux_data, train_list, valid_list, test_list):
        # basics
        self.recorder_bm = RecoderBestModel(save_dir)
        self.recorder_lm = RecorderLossAndMetrics(save_dir)
        self.recorder_cp = RecoderCheckpoint(save_dir, args.continue_train, model, optimizer,
                                             self.recorder_bm.get_best, self.recorder_bm.set_best)
        
        self.start_epoch = self.recorder_cp.start_epoch
        self.load_best_model = self.recorder_bm.load_best_model
        self.print_best_models = self.recorder_bm.print_best_models
        self.save_checkpoint = self.recorder_cp.save_checkpoint
        self.load_checkpoint = self.recorder_cp.load_checkpoint
        self.close_writer = self.recorder_lm.close_writer
        
        # advanced
        self.recorder_nn = RecorderInnerModel(save_dir, aux_data, train_list, valid_list, test_list)
        self.atth_record = args.record_atth     # default False
        self.model_record = args.record_model   # default 0
        
        
    def train_recording(self, records, epoch):
        """Record losses in log and tensorboard."""
        self.recorder_lm.loss_record(records, "train", epoch)
        records.clear()     # save space


    def valid_recording(self, records, epoch, split, model):
        """Record losses and metrics in log and tensorboard, save best model."""
        self.recorder_lm.loss_record(records, split, epoch)
        self.recorder_bm.print_best_models(split)
        mrrs = self.recorder_lm.metrics_reocrd(records, split, epoch)
        self.recorder_bm.save_best_model(model, split, mrrs, epoch)
        records.clear()


    def inner_recording(self, model, epoch):
        """Record what's happending inside the model."""
        if self.atth_record:
            self.recorder_nn.atth_record(model)
        elif self.model_record and (epoch - 1) % self.model_record == 0:
            self.recorder_nn.model_record(model, epoch)
        else:
            pass       
        