def config_args(args, config):
    for key in config:
        value = config[key]
        if type(value)==str:
            eval(r"args.{0}='{1}'".format(key, config[key]))
        else:
            eval("args.{0}={1}".format(key, config[key]))

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
