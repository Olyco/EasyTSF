from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE
exp_conf = dict(
    model_name="N_BEATS",
    dataset_name='ECL',

    norm_variable=False,

    hist_len=24,
    pred_len=12,

    max_epochs=3,

    learning_rate=0.001,
    log_interval=10,
    log_val_interval=1,
    log_gradient_flow=False,
    weight_decay=1e-2,
    stack_types=['generic'],
    num_blocks=[2],
    num_block_layers=[2],
    widths=[128],
    sharing=False,
    expansion_coefficient_lengths=[32],
    backcast_loss_ratio=0.3,
    loss=RMSE(),

    lr=0.001,
)