ECL_conf = dict(
    dataset_name='ECL',
    var_num=321,
    freq=60,
    data_split=[18412, 2632, 5260], #70/10/20
    var_cut=321,
)

ECL1_short_conf = dict(
    dataset_name='ECL1_short',
    var_num=1,
    freq=60,
    data_split=[1296, 108, 108], #85/7.5/7.5
    var_cut=1,
)

Weather_conf = dict(
    dataset_name='Weather',
    var_num=20,
    freq=60,
    data_split=[21024, 1314, 3942], #80/5/15
    var_cut=20,
)