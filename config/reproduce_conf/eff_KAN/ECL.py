exp_conf = dict(
    model_name="eff_KAN",
    dataset_name='ECL',

    hist_len=48,
    pred_len=12,

    max_epochs=20,

    layers_hidden=[48 * 321, 10, 12 * 321],
    grid=3,
    k=3,

    lr=0.01,
)