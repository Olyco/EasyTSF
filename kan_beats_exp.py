import argparse
import importlib
import importlib.util
import os
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import pandas as pd


import lightning.pytorch as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.utilities.model_summary import ModelSummary
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting import TimeSeriesDataSet, Baseline, NBeats, GroupNormalizer, MultiNormalizer, EncoderNormalizer

from easytsf.runner.data_runner import DataInterface
from easytsf.runner.exp_runner import LTSFRunner
from easytsf.util import load_module_from_path

from train import load_config
from easytsf.model.KAN_BEATS import KANBeats


def prepare_data(config):
    data_path = os.path.join(config['data_root'], "{}.{}".format(config['dataset_name'], config['file_format']))
    data = pd.read_csv(data_path)
    data = data.rename(columns={'date': 'timestamp'})
    variable = data.iloc[:, 1:config['var_cut'] + 1].to_numpy()

    concatenated_df = pd.DataFrame()
    for i in range(variable.shape[1]):
        ts = pd.DataFrame(data={'variable': variable.T[i], 'id': [i for var in range(variable.shape[0])]})
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

    train_dataloader = train_data.to_dataloader(train=True, batch_size=config['batch_size'], num_workers=config['num_workers'])
    val_dataloader = val_data.to_dataloader(train=False, batch_size=config['batch_size'], num_workers=config['num_workers'])
    test_dataloader = test_data.to_dataloader(train=False, batch_size=config['batch_size'], num_workers=config['num_workers'])

    return train_dataloader, val_dataloader, test_dataloader, train_data


def train_func(hyper_conf, conf):
    if hyper_conf is not None: # add training config
        for k, v in hyper_conf.items():
            conf[k] = v
    conf['exp_time'] = datetime.now(pytz.timezone('Europe/Moscow')).strftime("%d%m%y_%H%M")

    L.seed_everything(conf["seed"])
    save_dir = os.path.join(conf["save_root"], '{}_{}'.format(conf["model_name"], conf["dataset_name"]))
    
    if "use_wandb" in conf and conf["use_wandb"]:
        run_logger = WandbLogger(save_dir=save_dir, name=conf["exp_time"], version='seed_{}'.format(conf["seed"]))
    else:
        run_logger = CSVLogger(save_dir=save_dir, name=conf["exp_time"], version='seed_{}'.format(conf["seed"]))
    conf["exp_dir"] = os.path.join(save_dir, conf["exp_time"], 'seed_{}'.format(conf["seed"]))

    # if conf["model_name"] == "KAN":
    #     conf["ckpt_path"] = os.path.join(conf["exp_dir"], "model")

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
        TuneReportCheckpointCallback(
            {conf["val_metric"]: conf["val_metric"]}, save_checkpoints=False, on="validation_end")
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
            log_interval=conf['log_interval'],
            log_gradient_flow=conf['log_gradient_flow'],
            weight_decay=conf['weight_decay'],
            learning_rate=conf['learning_rate'],
            reduce_on_plateau_patience=conf['reduce_on_plateau_patience'],
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
            log_interval=conf['log_interval'],
            log_gradient_flow=conf['log_gradient_flow'],
            weight_decay=conf['weight_decay'],
            learning_rate=conf['learning_rate'],
            reduce_on_plateau_patience=conf['reduce_on_plateau_patience'],
        )
    print(model.hparams)

    print(ModelSummary(model, max_depth=-1))

    trainer.fit(
        model=model, 
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
        )
    # if conf["model_name"] == "KAN":
    #     model.model.saveckpt() # в файлы колаба

    trainer.test(model, dataloaders=test_dataloader, ckpt_path='best')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str) # experiment config
    parser.add_argument("-d", "--data_root", default="drive/MyDrive/VKR/Data/Time series", type=str, help="data root") # "drive/MyDrive/Оля/"
    parser.add_argument("-s", "--save_root", default="drive/MyDrive/VKR/Results/save", help="save root") # "drive/MyDrive/Оля/save"
    parser.add_argument("--devices", default='auto', type=str, help="device' id to use")
    parser.add_argument("--use_wandb", default=0, type=int, help="use wandb")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    args = parser.parse_args()

    training_conf = {
        "seed": int(args.seed),
        "data_root": args.data_root,
        "save_root": args.save_root,
        "devices": args.devices,
        "use_wandb": args.use_wandb,
    }
    init_exp_conf = load_config(args.config)
    train_func(training_conf, init_exp_conf)