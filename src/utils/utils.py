import torch
import math

import logging
import bitsandbytes as bnb

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(lineno)s",
)
from transformers import Qwen2VLForConditionalGeneration


def get_pred_index(pred_logits):
    # b,l,vocab_size
    index = torch.argmax(pred_logits, dim=-1)


def get_optimizer(model, train_args):
    if train_args.optimizer == "adam":
        weight_decay = train_args.weight_decay if train_args.weight_decay else 0
        # 0.9,0.95
        betas = (
            (train_args.beta1, train_args.beta2)
            if train_args.beta1 and train_args.beta2
            else (0.9, 0.999)
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=train_args.lr, weight_decay=weight_decay, betas=betas
        )
    elif train_args.optimizer == "adamw":
        weight_decay = train_args.weight_decay if train_args.weight_decay else 0
        # 0.9,0.95
        betas = (
            (train_args.beta1, train_args.beta2)
            if train_args.beta1 and train_args.beta2
            else (0.9, 0.999)
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=train_args.lr, weight_decay=weight_decay, betas=betas
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_args.lr)
    elif train_args.optimizer == "adamw8bit":
        # 需要 pip install bitsandbytes,
        # import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW(model.parameters(), lr=train_args.lr, optim_bits=8)
    elif train_args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=train_args.lr, momentum=train_args.momentum
        )
    else:
        raise ValueError("optimizer not supported")
    return optimizer


class Scheduler:
    """
    是否采用warmup, warmup_steps, 学习率衰减策略
    """

    def __init__(self, optimizer, train_args) -> None:
        self.train_args = train_args
        self.steps = 0
        self.optimizer = optimizer
        self.lrs = [param_group["lr"] for param_group in optimizer.param_groups]
        self.last_lr = None

    def get_lr(self):
        lrs = []
        if self.train_args.warm_up:
            # 线性预热
            if self.steps < self.train_args.warmup_steps:
                lr_min = self.train_args.lr_min
                lrs = [
                    self.steps * (lr - lr_min) / self.train_args.warmup_steps + lr_min
                    for lr in self.lrs
                ]
                self.last_lr = lrs
                return lrs
            elif self.train_args.scheduler == "cosine":
                # 余弦退火
                lrs = [
                    self.train_args.lr_min
                    + 0.5
                    * (lr - self.train_args.lr_min)
                    * (
                        1
                        + math.cos(
                            (self.steps - self.train_args.warmup_steps)
                            / (self.train_args.max_steps - self.train_args.warmup_steps)
                            * math.pi
                        )
                    )
                    for lr in self.lrs
                ]
                self.last_lr = lrs
            elif self.train_args.scheduler == "linear":
                # 线性衰减
                lrs = [
                    lr
                    - (self.steps - self.train_args.warmup_steps)
                    * (lr - self.train_args.lr_min)
                    / (self.train_args.max_steps - self.train_args.warmup_steps)
                    for lr in self.lrs
                ]
                self.last_lr = lrs
            else:
                raise ValueError("scheduler not supported")

        # 不采用预热 直接进行训练
        elif self.train_args.scheduler == "cosine":
            # 余弦退火
            lrs = [
                self.train_args.lr_min
                + 0.5
                * (lr - self.train_args.lr_min)
                * (
                    1
                    + math.cos(
                        (self.steps - self.train_args.warmup_steps)
                        / (self.train_args.max_steps)
                        * math.pi
                    )
                )
                for lr in self.lrs
            ]
            self.last_lr = lrs
        elif self.train_args.scheduler == "linear":
            # 线性衰减
            lrs = [
                lr
                - (self.steps - self.train_args.warmup_steps)
                * (lr - self.train_args.lr_min)
                / (self.train_args.max_steps - self.train_args.warmup_steps)
                for lr in self.lrs
            ]
            self.last_lr = lrs
        else:
            raise ValueError("scheduler not supported")

        return lrs

    def step(self, step):
        self.steps = step
        lrs = self.get_lr()
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = lrs[i]

    def reset(self):
        self.steps = 0
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lrs[i]

    def get_last_lr(self):
        return self.lrs


# from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR, CosineAnnealingWarmRestarts


def get_scheduler(optimizer, train_args):

    # if train_args.scheduler=='cosine':
    #     scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=train_args.max_steps,eta_min=train_args.lr_min)
    # elif train_args.scheduler=='linear':
    #     scheduler=torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=train_args.lr_min,total_iters=train_args.max_steps)
    # else:
    #     raise ValueError('scheduler not supported')

    return Scheduler(optimizer, train_args)


class EarlyStopping:
    def __init__(self, patience=5, delta=0) -> None:
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.best_epoch = None

    def step(self, loss, epoch):
        if self.best_loss - self.delta > loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

    def stopping(self):
        if self.patience <= self.counter:
            return True
        else:
            return False


class Evaluate:
    def __init__(self) -> None:
        pass

    @torch.no_grad()
    def perplexity(self, logits, labels, attention_mask, ignore_index):
        # TODO: 一个val的评价指标之一,需要注意的时padding,这些都是没有的，需要考虑将padding的损失去掉
        # 输入的就是初model return的logit,attention_mask :{b,l},ignore_index:{b,l},
        b, l, vocal_size = logits.shape
        # 全部都计算 概率
        # labels :{b,l}
        # labels = labels.unsqueeze(-1)
        logger.debug("________________________________________")
        logger.debug(f"the logits is {logits[0, 0]}")
        logger.debug(f"the labels is {labels[0, 0]}")
        preds = (
            torch.softmax(logits, dim=-1)
            .gather(dim=-1, index=labels.unsqueeze(-1))
            .squeeze(-1)
        )

        logger.debug(preds)
        logger.debug("#######################################################")
        logger.debug(f"the shape of logits is {logits.shape}")
        logger.debug(f"the shape of labels is {labels.shape}")
        logger.debug(f"the shape of preds is {preds.shape}")
        # 选择的那部分的概率

        mask = (attention_mask[..., 1:] == 1) == (ignore_index[..., 1:] != -100)
        logger.debug(f"the shape of mask is {mask.shape}")
        masked_preds = torch.where(mask, preds, torch.ones_like(preds))
        logger.debug(f"the shape of masked_preds is {masked_preds.shape}")
        perplexity = 2 ** (-torch.log2(masked_preds).sum() / torch.sum(mask))

        return perplexity

    @torch.no_grad()
    def rouge_score(self):
        # TODO: 另一个评价 val 的指标

        pass


def get_model(model_args):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name,
        torch_dtype=torch.float32,
        device_map=model_args.device_map,
        cache_dir=model_args.cache_dir,
    )
    return model
