exp_conf = dict(
    model_name="KAN",
    dataset_name='ETTh1',

    hist_len=96,
    pred_len=48,

    max_epochs=3,

    width=[96 * 7, 50, 50 , 48 * 7],
    grid=3,
    k=3,

    lr=0.01,

    save_root="drive/MyDrive/VKR/Results/save"
)
