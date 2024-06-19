import os
import numpy as np

# 定数
file = 100  # 各データセットに含まれるファイルの数

# MCEPデータを保持するクラス
class MCEPData:
    def __init__(self, name, word, frame, mcepdata):
        self.name = name  # データセットの名前
        self.word = word  # 対応する単語
        self.frame = frame  # フレーム数
        self.mcepdata = mcepdata  # MCEPデータ

def read_mcep_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} does not exist.")
        return None

    with open(filepath, 'r') as f:
        lines = f.readlines()
    name, word, frame = lines[0].strip(), lines[1].strip(), int(lines[2].strip())
    mcepdata = np.array([list(map(float, line.split())) for line in lines[3:3+frame]])
    return MCEPData(name, word, frame, mcepdata)

def dp_matching(temp_data, targ_data):
    d = np.linalg.norm(temp_data.mcepdata[:, None] - targ_data.mcepdata, axis=2)
    g = np.full_like(d, np.inf)
    g[0, 0] = d[0, 0]
    g[1:, 0] = np.cumsum(d[1:, 0])
    g[0, 1:] = np.cumsum(d[0, 1:])

    for i in range(1, temp_data.frame):
        for j in range(1, targ_data.frame):
            g[i, j] = d[i, j] + min(g[i-1, j], g[i, j-1], g[i-1, j-1])

    return g[-1, -1] / (temp_data.frame + targ_data.frame)

def main():
    print("data number = 11, 12, 21, 22")
    temp_num = int(input("set the template data: "))
    targ_num = int(input("set the target data: "))
    print(f"start voice recognition city{temp_num:03} and city{targ_num:03}")
    count = 0

    for h0 in range(file):
        temp_filepath = f"./city_mcepdata/city{temp_num:03}/city{temp_num:03}_{h0+1:03}.txt"
        temp_data = read_mcep_data(temp_filepath)

        distances = []

        for h in range(file):
            targ_filepath = f"./city_mcepdata/city{targ_num:03}/city{targ_num:03}_{h+1:03}.txt"
            targ_data = read_mcep_data(targ_filepath)

            distances.append(dp_matching(temp_data, targ_data))

        min_distance = min(distances)
        matched_file = distances.index(min_distance)
        if matched_file == h0:
            count += 1
        else:
            print(f"===========Result NOT Matching===========\n"
                  f"temp_data: city{temp_num:03}_{h0+1:03}.txt\n"
                  f"target_data: city{targ_num:03}_{matched_file+1:03}.txt\n"
                  f"tangokankyori: {min_distance}")

    accuracy = (count / file) * 100
    FNAME_OUTPUT = f"./results/city{temp_num:03}_vs_city{targ_num:03}_result.txt"
    os.makedirs(os.path.dirname(FNAME_OUTPUT), exist_ok=True)

    # with open(FNAME_OUTPUT, 'a') as f_output:
    #     f_output.write(f"正答率{accuracy:.2f}%です。\n")
    print(f"\naccuracy:{accuracy:.2f}% ")

if __name__ == "__main__":
    main()
