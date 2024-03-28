class CFG:
    seed = 42
    num_frames = 4
    num_classes = 2
    batch_size = 4
    num_workers = 0 #32
    learning_rate = 1e-4
    num_epochs = 100
    drop_prob = 0.
    dynamic_frames = True
    pretrained = True
    height = 224
    width = 224
    slowfast = True
    limit_step_per_batch=200
