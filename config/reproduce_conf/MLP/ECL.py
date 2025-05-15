exp_conf = dict(
    model_name="MLP",
    dataset_name='ECL',

    hist_len=24,
    pred_len=6,

    max_epochs=5,

    width=[24 * 321, 5, 6 * 321],
    
    lr=0.001,
)
