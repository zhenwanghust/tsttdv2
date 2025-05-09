from Train_eval import train, eval, select_best,select_params
# 代码库中提供了五个数据集中已训练好的权重，如果要验证本文中报告的结果，需要修改如下配置：
# state选择eval，band按照对应的数据集进行选择，path 和 name改为对应数据集的名字（name不含后缀），test_load_weight 改为finalbest。直接运行python Main.py即可

# 如果要复现训练出本文中报告的结果，需要修改如下配置：
# state选择train，band按照对应的数据集进行选择，path 和 name改为对应数据集的名字（name不含后缀），运行python Main.py
# train结束后，修改state为eval，修改test_load_weight为best_model_0_98_0_98_pt，再次运行python Main.py即可


def main(model_config=None):

    modelConfig = {
        "state": "eval",  # train, eval, select_params
        #eval之前需要改save path！！！
        #change proportion in DATA.PY
        "epoch": 40,
        "band": 189,  # 204 for urban1（即为Texas Coast）
        # 191 for urban3（即为Gainesville）
        #189 for indian Pine
        # 189 for Sandiego / Sandiego 2
        "multiplier": 2,
        "seed": 1,
        "batch_size": 256,
        "group_length": 20,# 组嵌入分组长度
        "depth": 4,#transformer的叠加个数
        "heads": 4,#已经被删了
        "dim_head": 64,#已经被删了
        "mlp_dim": 64,
        "adjust": False,
        "channel": 128,
        "lr": 1e-4,
        "epision": 5,
        "grad_clip": 1.,
        "device": "cuda:0",
        "training_load_weight": None,
        "save_dir": "./Checkpoint/",
        "test_load_weight":"finalbest",
        "path": "Sandiego.mat",
        "name":"Sandiego",
        "alphamax":0.35,
        "betamax":0.10,
        #finalbest

    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
        _=select_best(modelConfig)
    elif modelConfig["state"] == "eval":
        eval(modelConfig)
    elif modelConfig["state"] == "select_params":
        select_params(modelConfig)
    else:
        _=select_best(modelConfig)

if __name__ == '__main__':
    main()
