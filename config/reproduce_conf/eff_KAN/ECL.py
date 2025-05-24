exp_conf = dict(
    model_name="eff_KAN",
    dataset_name='ECL',

    hist_len=24,
    pred_len=6,

    max_epochs=5,

    layers_hidden=[24 * 321, 5, 6 * 321],
    grid=3,#
    k=3,#

    lr=0.001,
)