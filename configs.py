def config_args(args, config):
    if "data_dir" in config:
        args.data_dir = config["data_dir"]
    if "save_dir" in config:
        args.save_dir = config["save_dir"]
    if "nr_filters" in config:
        args.nr_filters = config["nr_filters"]
    if "nr_resnet" in config:
        args.nr_resnet = config["nr_resnet"]
    if "data_set" in config:
        args.data_set = config["data_set"]
    if "batch_size" in config:
        args.batch_size = config["batch_size"]
    if "init_batch_size" in config:
        args.init_batch_size = config["init_batch_size"]



configs = {}

configs['test'] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/save-backward-rename",
    "nr_filters": 160,
    "nr_resnet": 5,
    "data_set": "celeba",
    "batch_size": 6,
    "init_batch_size": 6,
    "nr_gpu": 8,
}
