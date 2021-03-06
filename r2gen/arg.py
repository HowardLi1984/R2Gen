class arg():
    def __init__(self):
        self.image_dir = 'data/mimic_cxr/images/'
        self.ann_path = 'data/mimic_cxr/annotation.json'
        self.dataset_name = 'mimic_cxr'
        self.max_seq_length = 100
        self.threshold = 10
        self.num_workers = 2
        self.batch_size = 16
        self.visual_extractor = 'resnet101'
        self.visual_extractor_pretrained = True
        self.d_model = 512
        self.d_ff = 512
        self.d_vf = 2048
        self.num_heads = 8
        self.num_layers = 3
        self.dropout = 0.1
        self.logit_layers = 1
        self.bos_idx = 0
        self.eos_idx = 0
        self.pad_idx = 0
        self.use_bn = 0
        self.drop_prob_lm = 0.5
        self.rm_num_slots = 3
        self.rm_num_heads = 8
        self.rm_d_model = 512
        self.sample_method = 'beam_search'
        self.beam_size = 3
        self.temperature = 1.0
        self.sample_n = 1
        self.group_size = 1
        self.output_logsoftmax = 1
        self.decoding_constraint = 0
        self.block_trigrams = 1
        self.n_gpu = 1
        self.epochs = 30
        self.save_dir = 'results/mimic_cxr'
        self.record_dir = 'records/'
        self.save_period = 1
        self.monitor_mode = 'max'
        self.monitor_metric = 'BLEU_4'
        self.early_stop = 50
        self.optim = 'Adam'
        self.lr_ve = 5e-5
        self.lr_ed = 1e-4
        self.weight_decay = 5e-5
        self.amsgrad = True
        self.lr_scheduler = 'StepLR'
        self.step_size = 1
        self.gamma = 0.8
        self.seed = 456789
        self.resume = ''
