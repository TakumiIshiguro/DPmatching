import os
import numpy as np

file = 100  # 各データセットに含まれるファイルの数
result_dir = './RESULT'
os.makedirs(result_dir, exist_ok=True)

class MCEPData:
    def __init__(self, name, word, frame, mcepdata):
        self.name = name  
        self.word = word  
        self.frame = frame  
        self.mcepdata = mcepdata  

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
    d = np.zeros((temp_data.frame, targ_data.frame))
    g = np.zeros((temp_data.frame, targ_data.frame))

    for i in range(temp_data.frame):
        for j in range(targ_data.frame):
            d[i, j] = np.sum((temp_data.mcepdata[i] - targ_data.mcepdata[j]) ** 2)
            d[i, j] = np.sqrt(d[i, j])

    g[0, 0] = d[0, 0]
    for i in range(1, temp_data.frame):
        g[i, 0] = g[i - 1, 0] + d[i, 0]
    for j in range(1, targ_data.frame):
        g[0, j] = g[0, j - 1] + d[0, j]

    for i in range(1, temp_data.frame):
        for j in range(1, targ_data.frame):
            g[i, j] = min(
                g[i, j - 1] + d[i, j],       
                g[i - 1, j - 1] + 2 * d[i, j], 
                g[i - 1, j] + d[i, j]        
            )

    return g[temp_data.frame - 1, targ_data.frame - 1] / (temp_data.frame + targ_data.frame)

def print_progress(current, total):
    percentage = (current / total) * 100
    bar_length = 100
    progress = int(bar_length * (percentage / 100.0))
    bar = '#' * progress + ' ' * (bar_length - progress)
    print(f"\r{percentage:.0f}% [{bar}]", end="", flush=True)

def main():
    print("dataset number = 11, 12, 21, 22")
    temp_num = int(input("set the template dataset: "))
    targ_num = int(input("set the target dataset: "))
    print(f"start voice recognition city{temp_num:03} and city{targ_num:03}")
    result_file = f"{result_dir}/city{temp_num:03}_city{targ_num:03}.txt"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, 'a') as f_output:
        f_output.write("----------result----------")
    count = 0

    for i in range(file):
        temp_filepath = f"./city_mcepdata/city{temp_num:03}/city{temp_num:03}_{i+1:03}.txt"
        temp_data = read_mcep_data(temp_filepath)

        distances = []
        print(f"\nfile: city{temp_num:03}_{i+1:03}")

        for j in range(file):
            targ_filepath = f"./city_mcepdata/city{targ_num:03}/city{targ_num:03}_{j+1:03}.txt"
            targ_data = read_mcep_data(targ_filepath)

            distances.append(dp_matching(temp_data, targ_data))

            print_progress(j + 1, file)

        min_distance = min(distances)
        matched_file = distances.index(min_distance)

        if matched_file == i:
            count += 1
        else:
            error_string = (
                f"\nNOT Matching \n<info>"
                f"\ntemplate_data: city{temp_num:03}_{i+1:03}.txt"
                f"\ntarget_data: city{targ_num:03}_{matched_file+1:03}.txt"
                f"\ndistance: {min_distance}"
            )
            print(error_string)
            with open(result_file, 'a') as f_output:
                f_output.write(error_string)

    accuracy = (count / file) * 100
    print(f"\ncity{temp_num:03} and city{targ_num:03} matching accuracy: {accuracy}% ")
    with open(result_file, 'a') as f_output:
        f_output.write("\n" + "accuracy = " + str(accuracy))

if __name__ == "__main__":
    main()
