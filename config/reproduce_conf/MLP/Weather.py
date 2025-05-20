exp_conf = dict(
    model_name="MLP",
    dataset_name='Weather',

    hist_len=24,
    pred_len=6,

    max_epochs=50,

    width=[24 * 20, 5, 6 * 20],
    
    lr=0.0001,
)
