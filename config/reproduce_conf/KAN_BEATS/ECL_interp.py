from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE

exp_conf = dict(
    model_name="KAN_BEATS",
    dataset_name='ECL',
    var_cut=10,
    data_split=[21043, 2630, 2631],

    norm_variable=True,
    batch_sampler='synchronized',

    hist_len=48,
    pred_len=12,

    max_epochs=3,

    grid_size=3,
    spline_order=3,

    stack_types=['trend','seasonality'],
    num_blocks=[2, 2],
    num_block_layers=[4, 4],
    widths=[16, 16],
    sharing=False,
    expansion_coefficient_lengths=[3, 12],
    backcast_loss_ratio=0.1,#
    loss=MAE(),

    val_metric="val_loss",
    test_metric="test_mae",
    batch_size=64,

    log_interval=10,
    # log_val_interval=1,
    log_gradient_flow=False,
    weight_decay=1e-2,

    learning_rate=0.0001,
    lr=0.0001,
    reduce_on_plateau_patience=10,
)