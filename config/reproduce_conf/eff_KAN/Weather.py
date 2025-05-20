exp_conf = dict(
    model_name="eff_KAN",
    dataset_name='Weather',

    hist_len=24,
    pred_len=6,

    max_epochs=50,

    layers_hidden=[24 * 20, 2, 6 * 20],
    grid=3,
    k=3,

    lr=0.000001,
)