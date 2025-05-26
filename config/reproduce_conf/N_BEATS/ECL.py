from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE
from .....EasyTSF.easytsf.model.KAN_BEATS import customR2Score
from torch import nn

exp_conf = dict(
    model_name="N_BEATS",
    dataset_name='ECL',
    var_cut=10,

    norm_variable=True,
    batch_sampler='synchronized',

    hist_len=24,
    pred_len=12,

    max_epochs=3,

    stack_types=['generic'],
    num_blocks=[2],
    num_block_layers=[2],
    widths=[64],
    sharing=False,
    expansion_coefficient_lengths=[32],
    backcast_loss_ratio=0.1,
    loss=MAE(),
    logging_metrics=nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE(), customR2Score()]),

    val_metric="val_loss",
    test_metric="test_mae",
    batch_size=128,

    log_interval=10,
    # log_val_interval=1,
    log_gradient_flow=False,
    weight_decay=1e-2,

    lr=0.001,
    learning_rate=0.001,
    reduce_on_plateau_patience=10,
)