import os


class BaseConfig:
    # ---------------- model config ---------------- #
    num_classes = 2
    depth = 1.00
    width = 1.00
    act = 'silu'

    # ---------------- dataloader config ---------------- #
    # set worker to 4 for shorter dataloader init time
    data_num_workers = 4
    input_size = (640, 640)  # (height, width)
    # Actual multiscale ranges: [640-5*32, 640+5*32].
    # To disable multiscale training, set the
    # self.multiscale_range to 0.
    multiscale_range = 5
    # You can uncomment this line to specify a multiscale range
    # self.random_size = (14, 26)
    data_dir = '../data/'
    train_ann = "annotations.json"
    val_ann = "annotations.json"

    # --------------- transform config ----------------- #
    mosaic_prob = 1.0
    mixup_prob = 1.0
    hsv_prob = 1.0
    flip_prob = 0.5
    degrees = 10.0
    translate = 0.1
    mosaic_scale = (0.1, 2)
    mixup_scale = (0.5, 1.5)
    shear = 2.0
    enable_mixup = True

    # --------------  training config --------------------- #
    warmup_epochs = 5
    # max_epoch = 300
    warmup_lr = 0
    basic_lr_per_img = 0.01 / 64.0
    scheduler = "yoloxwarmcos"
    no_aug_epochs = 15
    min_lr_ratio = 0.05
    ema = True

    weight_decay = 5e-4
    momentum = 0.9
    print_interval = 10
    eval_interval = 10
    exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    # -----------------  testing config ------------------ #
    test_size = (640, 640)
    test_conf = 0.01
    nmsthre = 0.65