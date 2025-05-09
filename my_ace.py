
def delete_target(result,data):

    # 执行上述代码
    mask = result < 0.5
    selected_vectors = data[mask]
    result_array = selected_vectors.reshape(-1, selected_vectors.shape[-1])
    print(data.shape)
    print("result_array 的形状:", result_array.shape)