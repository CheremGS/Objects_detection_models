from easydict import EasyDict


Config = EasyDict()

Config.CLASSES_NAME = ("T-72", "Tiger", "Kamaz_kung", "BMP-3", "MTLBu", "KRAZ_FUEL", "BTR-82A", "UAZ", "URAL_gruz")


# backbone
Config.slug = 'r50'
Config.fpn = False
Config.freeze_bn = False

# decoder
Config.bn_momentum = 0.1

# head
Config.head_channel = 16

# loss
Config.regr_loss = 'iou'
Config.loss_alpha = 1.
Config.loss_beta = 0.1
Config.loss_gamma = 1.

# dataset
Config.num_classes = 9
Config.batch_size = 1  # 128
Config.root = r"C:\Users\ITC-Admin\PycharmProjects\Detection_military\datasets\fixed_kstovo_new"
Config.split = 'trainval'
Config.resize_size = [512, 512]
Config.num_workers = 4
Config.mean = [0.40789654, 0.44719302, 0.47026115]
Config.std = [0.28863828, 0.27408164, 0.27809835]

# train
Config.optimizer = 'AdamW'
Config.lr = 1e-3
Config.AMSGRAD = True

Config.max_iter = 45000
Config.lr_schedule = 'WarmupMultiStepLR'
Config.gamma = 0.1
Config.steps = (35000, 40000)
Config.warmup_iters = 1000

Config.apex = False

# other
Config.gpu = True
Config.eval = False
Config.resume = False
Config.score_th = 0.1
Config.down_stride = 4
Config.topK = 100
Config.log_dir = './log'
Config.checkpoint_dir = './ckp'
Config.log_interval = 20