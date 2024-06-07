import torch
import torchmetrics
from typing import List, Dict, Any, Optional
from pytorch_lightning.loggers import Logger

class MetricAggregator:
    def __init__(self, num_classes: Optional[int] = None, phases: List[str] = ["train"], metrics: List[str] = ["acc", "cm", "f1"], device: str = "cpu", loggers: Optional[List[Logger]] = None) -> None:
        self.metrics = metrics
        self.aggregators: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        self.device = device
        self.num_classes = num_classes
        self.step_num = 1
        self.loggers = loggers if loggers is not None else []
        for phase in phases:
            self.aggregators[phase] = {}
            self.results[phase] = {}

    def init_agg(self, phase: str = "train", metric: str = "acc") -> None:
        if phase not in self.aggregators:
            self.aggregators[phase] = {}
            self.results[phase] = {}
        if metric == "acc":
            if self.num_classes is not None:
                self.aggregators[phase][metric] = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)
        elif metric == "cm":
            if self.num_classes is not None:
                self.aggregators[phase][metric] = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes).to(self.device)
        elif metric == "f1":
            if self.num_classes is not None:
                self.aggregators[phase][metric] = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes).to(self.device)
        else:
            self.aggregators[phase][metric] = torchmetrics.MeanMetric().to(self.device)

    def step(self, y_hat_prob: torch.Tensor, y: torch.Tensor, phase: str = "train", **kwargs: Any) -> None:
        if phase not in self.aggregators:
            self.aggregators[phase] = {}
            self.results[phase] = {}
        for metric in list(kwargs.keys()) + self.metrics:
            if metric not in self.aggregators[phase]:
                self.init_agg(phase=phase, metric=metric)
        for metric in ["acc", "cm", "f1"]:
            if metric in self.metrics:
                self.aggregators[phase][metric].update(y_hat_prob.to(self.device), y.to(self.device))
        for k, v in kwargs.items():
            if k in self.aggregators[phase]:
                self.aggregators[phase][k].update(v.to(self.device))
        for logger in self.loggers:
            logger.log_metrics({f"{phase}_{k}_step":v.detach().cpu().tolist() for k,v in kwargs.items()}, step=self.step_num)
        self.step_num+=1

    def compute(self, phase: str = "train") -> Dict[str, Any]:
        for k in self.aggregators[phase].keys():
            self.results[phase][f"{phase}_{k}_epoch"] = self.aggregators[phase][k].compute().detach().cpu().tolist()
        for logger in self.loggers:
            logger.log_metrics({f"{phase}_{k}_epoch":self.results[phase][f"{phase}_{k}_epoch"] for k in self.aggregators[phase].keys()}, step=self.step_num)
        return self.results[phase]

    def reset(self, phase: str = "train") -> None:
        for k in self.aggregators[phase].keys():
            self.aggregators[phase][k].reset()
