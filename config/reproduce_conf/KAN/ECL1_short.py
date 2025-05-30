exp_conf = dict(
    model_name="KAN",
    dataset_name='ECL1_short',

    hist_len=24,
    pred_len=12,

    max_epochs=15,

    width=[24, 5, 12],
    grid=3,
    k=3,

    lr=0.001,
    symbolic_enabled=False,
    auto_save=True,

    # ckpt_path= "drive/MyDrive/VKR/Results/save/model"
)