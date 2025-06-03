from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, MultiHorizonMetric
# from .....EasyTSF.easytsf.model.KAN_BEATS import customR2Score
from torch import nn
from torcheval.metrics import R2Score

class customR2Score(MultiHorizonMetric):

    def loss(self, y_pred, target):
        metric = R2Score(multioutput="uniform_average")
        metric.update(self.to_prediction(y_pred), target)
        r2 = metric.compute()
        return r2

exp_conf = dict(
    model_name="N_BEATS",
    dataset_name='ECL',
    var_cut=5,
    data_split=[21043, 2630, 2631],

    norm_variable=False,
    batch_sampler='synchronized',

    hist_len=48,
    pred_len=12,

    max_epochs=10,

    stack_types=["trend", "seasonality", "generic"],
    num_blocks=[4, 4, 4],
    num_block_layers=[4, 4, 4],
    widths=[100, 100, 100],
    sharing=False,
    expansion_coefficient_lengths=[3, 24, 32],
    backcast_loss_ratio=0.1,
    loss=MAE(),
    logging_metrics=nn.ModuleList([SMAPE(), MAE(), RMSE(), MAPE(), MASE(), customR2Score()]),

    val_metric="val_loss",
    test_metric="test_mae",
    batch_size=64,

    log_interval=10,
    # log_val_interval=1,
    log_gradient_flow=False,
    weight_decay=1e-2,

    lr=1e-8,
    learning_rate=1e-8,
    reduce_on_plateau_patience=3,#
    reduce_on_plateau_min_lr=1e-12,
)