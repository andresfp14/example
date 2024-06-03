class EarlyStoppingCustom(object):
    def __init__(self, monitor="valid_acc_epoch", min_delta=0.001, patience=20, mode="max", **kwargs):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.should_stop = False
        self.count = 0
        self.best_score = None
    
    def __call__(self, epoch_metrics):
        metric = epoch_metrics[self.monitor]
        if self.best_score is None:
            self.best_score = metric
            self.count = 0
        else:
            delta = (metric-self.best_score)/max(abs(self.best_score),0.000000001) 
            if (self.mode == "max") and (delta>self.min_delta):
                self.best_score = metric
                self.count = 0
            elif (self.mode == "min") and (-delta>self.min_delta):
                self.best_score = metric
                self.count = 0
            else:
                self.count += 1
            if self.count >= self.patience:
                self.should_stop = True