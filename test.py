import argparse
import importlib
import importlib.util
import os
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import pandas as pd
import time
import sys

import torch
import lightning.pytorch as L
import torch.optim.lr_scheduler as lrs
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.utilities.model_summary import ModelSummary
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting import TimeSeriesDataSet, Baseline, NBeats, GroupNormalizer, MultiNormalizer, EncoderNormalizer
from sklearn.preprocessing import MinMaxScaler
from lightning.pytorch.tuner import Tuner

from train import load_config
from easytsf.model.KAN_BEATS import KANBeats


def load_module_from_path(module_name, exp_conf_path):
    spec = importlib.util.spec_from_file_location(module_name, exp_conf_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_config(exp_conf_path):
    exp_conf = load_module_from_path("exp_conf", exp_conf_path).exp_conf

    task_conf_module = importlib.import_module('config.base_conf.task')
    task_conf = task_conf_module.task_conf

    data_conf_module = importlib.import_module('config.base_conf.datasets')
    data_conf = eval('data_conf_module.{}_conf'.format(exp_conf['dataset_name']))

    fused_conf = {**task_conf, **data_conf}
    fused_conf.update(exp_conf)

    return fused_conf


def prepare_data(config):
    data_path = os.path.join(config['data_root'], "{}.{}".format(config['dataset_name'], config['file_format']))
    data = pd.read_csv(data_path)
    data = data.rename(columns={'date': 'timestamp'})
    variable = data.iloc[:, 1:config['var_cut'] + 1].to_numpy()

    scaler = MinMaxScaler()
    if config['norm_variable']:
        scaler.fit(variable[:config['data_split'][0]])

        var = scaler.transform(variable)
    else:
      var = variable

    concatenated_df = pd.DataFrame()
    for i in range(var.shape[1]):
        ts = pd.DataFrame(data={'variable': var.T[i], 'id': [i for _ in range(var.shape[0])]})
        concatenated_df = pd.concat([concatenated_df, ts])
    concatenated_df = concatenated_df.reset_index()

    train_cut = concatenated_df.loc[concatenated_df['index'] < config['data_split'][0]]
    print(train_cut)

    val_cut = concatenated_df.loc[concatenated_df['index'].isin(range(config['data_split'][0], config['data_split'][0] + config['data_split'][1]))]
    print(val_cut)

    test_cut = concatenated_df.loc[concatenated_df['index'] >= config['data_split'][0] + config['data_split'][1]]
    print(test_cut)

    train_data = TimeSeriesDataSet(
        data=train_cut,
        time_idx="index",
        target="variable",
        categorical_encoders={"id": NaNLabelEncoder().fit(train_cut.id)},
        group_ids=['id'],
        max_encoder_length=config['hist_len'],
        max_prediction_length=config['pred_len'],
        time_varying_unknown_reals=["variable"],
    )
    val_data = TimeSeriesDataSet.from_dataset(train_data, val_cut)
    test_data = TimeSeriesDataSet.from_dataset(train_data, test_cut)

    train_dataloader = train_data.to_dataloader(
        train=True, 
        batch_size=config['batch_size'], 
        num_workers=config['num_workers'], 
        batch_sampler=config['batch_sampler'],
    )
    val_dataloader = val_data.to_dataloader(
        train=False, 
        batch_size=config['batch_size'], 
        num_workers=config['num_workers'],
        )
    test_dataloader = test_data.to_dataloader(
        train=False, 
        batch_size=config['batch_size'], 
        num_workers=config['num_workers'],
    )

    return train_dataloader, val_dataloader, test_dataloader, train_data


def train_func(hyper_conf, conf):
    if hyper_conf is not None: # add training config
        for k, v in hyper_conf.items():
            conf[k] = v
    conf['exp_time'] = datetime.now(pytz.timezone('Europe/Moscow')).strftime("%d%m%y_%H%M")

    L.seed_everything(conf["seed"])
    save_dir = os.path.join(conf["save_root"], '{}_{}'.format(conf["model_name"], conf["dataset_name"]))
    
    run_logger = CSVLogger(save_dir=save_dir, name=conf["exp_time"], version='seed_{}'.format(conf["seed"]))
    conf["exp_dir"] = os.path.join(save_dir, conf["exp_time"])

    callbacks = [
        ModelCheckpoint(
            monitor=conf["val_metric"],
            mode="min",
            save_top_k=1,
            save_last=False,
            every_n_epochs=1,
        ),
        EarlyStopping(
            monitor=conf["val_metric"],
            mode='min',
            patience=conf["es_patience"],
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = L.Trainer(
        devices=conf["devices"],
        precision=conf["precision"] if "precision" in conf else "32-true",
        logger=run_logger,
        callbacks=callbacks,
        max_epochs=conf["max_epochs"],
        gradient_clip_algorithm=conf["gradient_clip_algorithm"] if "gradient_clip_algorithm" in conf else "norm",
        gradient_clip_val=conf["gradient_clip_val"],
        default_root_dir=conf["save_root"],
        deterministic=True,
    )

    train_dataloader, val_dataloader, test_dataloader, train_data = prepare_data(conf)

    if conf['model_name'] == "KAN_BEATS":
        model = KANBeats.from_dataset(
            train_data,
            grid_size=conf['grid_size'],
            spline_order=conf['spline_order'],
            stack_types=conf['stack_types'],
            num_blocks=conf['num_blocks'],
            num_block_layers=conf['num_block_layers'],
            widths=conf['widths'],
            sharing=conf['sharing'],
            expansion_coefficient_lengths=conf['expansion_coefficient_lengths'],
            backcast_loss_ratio=conf['backcast_loss_ratio'],
            loss=conf['loss'],
            logging_metrics=conf['logging_metrics'],
            optimizer=conf['optimizer'],
            log_interval=3,
            log_gradient_flow=conf['log_gradient_flow'],
            weight_decay=conf['weight_decay'],
            learning_rate=conf['learning_rate'],
            reduce_on_plateau_patience=2,
            reduce_on_plateau_min_lr=conf['reduce_on_plateau_min_lr'],
            reduce_on_plateau_reduction=10.0,
            )
    elif conf['model_name'] == "N_BEATS":
        model = NBeats.from_dataset(
            train_data,
            stack_types=conf['stack_types'],
            num_blocks=conf['num_blocks'],
            num_block_layers=conf['num_block_layers'],
            widths=conf['widths'],
            sharing=conf['sharing'],
            expansion_coefficient_lengths=conf['expansion_coefficient_lengths'],
            backcast_loss_ratio=conf['backcast_loss_ratio'],
            loss=conf['loss'],

            logging_metrics=conf['logging_metrics'],
            optimizer=conf['optimizer'],
            log_interval=3,
            log_gradient_flow=conf['log_gradient_flow'],
            weight_decay=conf['weight_decay'],
            learning_rate=conf['learning_rate'],
            reduce_on_plateau_patience=2,
            reduce_on_plateau_min_lr=conf['reduce_on_plateau_min_lr'],
            reduce_on_plateau_reduction=10.0,
        )

    # print(ModelSummary(model, max_depth=-1))

    #
    res = Tuner(trainer).lr_find(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, min_lr=1e-3)
    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=False, suggest=True)
    fig.savefig(f"{conf['model_name']}_{conf['exp_time']}_lr_{res.suggestion():.010f}.png")
    model.hparams.learning_rate = res.suggestion()
    
    #change lr everywhere
    for g in trainer.optimizers[0].param_groups:
      g['lr'] = model.hparams.learning_rate
    lr_state = trainer.lr_scheduler_configs[0].scheduler.state_dict()
    lr_state['_last_lr'] = model.hparams.learning_rate
    trainer.lr_scheduler_configs[0].scheduler.load_state_dict(lr_state)

    print(model.hparams)
    print(trainer.optimizers[0])
    print(trainer.lr_scheduler_configs[0].scheduler.state_dict())
    #

    start = time.time()
    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
        )
    end = time.time()
    train_time = end - start
    print(f"Training time: {train_time:.03f} s ({(train_time / 60):.03f} min, {(train_time / 3600):.03f} h)")

    trainer.test(model, dataloaders=test_dataloader, ckpt_path='best')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str) # experiment config
    parser.add_argument("-d", "--data_root", default="", type=str, help="data root") # папка с данными
    parser.add_argument("-s", "--save_root", default="save", help="save root") # папка сохранения
    parser.add_argument("--devices", default='auto', type=str, help="device' id to use")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    args = parser.parse_args()

    training_conf = {
        "seed": int(args.seed),
        "data_root": args.data_root,
        "save_root": args.save_root,
        "devices": args.devices,
    }
    init_exp_conf = load_config(args.config)
    train_func(training_conf, init_exp_conf)