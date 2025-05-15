exp_conf = dict(
    model_name="MLP",
    dataset_name='ECL',

    hist_len=48,
    pred_len=12,

    max_epochs=5,

    width=[48 * 321, 10, 12 * 321],
    
    lr=0.01,

    save_root="drive/MyDrive/VKR/Results/save"
)
