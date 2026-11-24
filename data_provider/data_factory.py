# Adapted from RAFT (https://github.com/archon159/RAFT) and Time-Series-Library
# (https://github.com/thuml/Time-Series-Library), both MIT License.
from torch.utils.data import DataLoader

from data_provider.data_loader import Dataset_Custom


def data_provider(args, flag):
    """Return (dataset, loader) for flag in {'train', 'val', 'test'}."""
    data_set = Dataset_Custom(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.pred_len],
        features=args.features,
        target=args.target,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=(flag != 'test'),
        num_workers=args.num_workers,
        drop_last=False)
    return data_set, data_loader
