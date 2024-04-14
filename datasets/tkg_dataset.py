"""Dataset class for loading and processing TKG datasets."""
import logging
import os

from ._knowledge_graph import load_from_local
from ._utils import load_all_answers_for_time_filter, split_by_time


class TKGDataset(object):
    """Temporal Knowledge Graph dataset class."""

    def __init__(self, dataset, debug):
        """Creates TKG dataset object for data loading.

        Args:
             dataset: String indicating the dataset to use
        """
        data_path = os.environ["DATA_PATH"]
        data = load_from_local(data_path, dataset)
        self.debug = debug
        self.debug_len = 10
        self.num_ent = data.num_nodes
        self.num_rel = data.num_rels
              
        self.triplelst_dict = {}
        self.ans4tflst_dict = {}
        for split in ["train", "valid", "test"]:
            data_split = split_by_time(data.splits[split])     # train/valid/test list
            self.triplelst_dict[split] = data_split
            all_ans_list_split = load_all_answers_for_time_filter(
                                data.splits[split], self.num_rel, self.num_ent, False)
            self.ans4tflst_dict[split] = all_ans_list_split
        
        if self.debug: logging.info(f"Debugging Length Set to {self.debug_len} ")


    def get_snaps(self, split):
        """Get a list containing all snapshots (np.array: num_event x 3) in a split.
        
        Args: 
            split: String indicating the split to use (train/valid/test)
            
        Returns:
            snaps: List containing all snaps in a split
        """
        snaps = self.triplelst_dict[split] if not self.debug else self.triplelst_dict[split][:self.debug_len]
        return snaps


    def get_ans4tf(self, split):
        """Get all ent answers of a split for time filter to compute ranking metrics (for testing)."""
        return self.ans4tflst_dict[split] if not self.debug else self.ans4tflst_dict[split][:self.debug_len]

    def len_time(self, split):
        """Get total time length of snapshots in a split."""
        return len(self.triplelst_dict[split]) if not self.debug else self.debug_len
    
    def get_shape(self):
        """Returns TKG dataset shape (num_ent, num_rel)."""
        return self.num_ent, self.num_rel