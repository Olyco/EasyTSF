exp_conf = dict(
    model_name="eff_KAN",
    dataset_name='ETTh1',

    hist_len=48,
    pred_len=12,

    max_epochs=100,

    layers_hidden=[48 * 7, 100 , 12 * 7],
    grid=3,
    k=3,

    lr=0.01,
)