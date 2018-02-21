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
    if "spatial_conditional" in config:
        args.spatial_conditional = config['spatial_conditional']
    if "global_conditional" in config:
        args.global_conditional = config['global_conditional']
    if "save_interval" in config:
        args.save_interval = config['save_interval']



configs = {}

configs['cifar'] = {
    "data_dir": "/data/ziz/not-backed-up/jxu/cifar",
    "save_dir": "/data/ziz/jxu/models/cifar-test",
    "nr_filters": 30,
    "nr_resnet": 2,
    "data_set": "cifar",
    "batch_size": 8,
    "init_batch_size": 8,
    "spatial_conditional": True,
    "save_interval": 2,
}
