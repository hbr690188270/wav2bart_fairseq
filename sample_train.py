import numpy as np

file_path = "/data/bairu/audio_dataset/LibriSpeech/processed_file/"

audio_name_list = []
with open(file_path + "train.tsv", 'r', encoding = 'utf-8') as f:
    prefix = f.readline().strip()
    for line in f:
        audio_name_list.append(line.strip())

with open(file_path + "train.ltr", 'r', encoding = 'utf-8') as f:
    ltr_list = []
    for line in f:
        ltr_list.append(line.strip())

print(audio_name_list[0])
print(ltr_list[0])
assert len(audio_name_list) == len(ltr_list)
rand_idx_1 = np.random.choice(len(audio_name_list),1, replace = False)
rand_idx_10 = np.random.choice(len(audio_name_list),10, replace = False)
rand_idx_100 = np.random.choice(len(audio_name_list),100, replace = False)
rand_idx_2000 = np.random.choice(len(audio_name_list),2000, replace = False)
rand_idx_10000 = np.random.choice(len(audio_name_list),10000, replace = False)

# for num in (1,10,100,2000,10000):
for num in [10000]:
    select_idx_list = eval("rand_idx_%d"%(num))
    write_audio_name_list = [prefix]
    write_ltr_list = []
    for idx in select_idx_list:
        curr_audio_name = audio_name_list[idx]
        curr_ltr = ltr_list[idx]
        write_audio_name_list.append(curr_audio_name)
        write_ltr_list.append(curr_ltr)
    
    with open(file_path + "train_%d.tsv"%(num),'w',encoding = 'utf-8') as f:
        for item in write_audio_name_list:
            f.write(item + '\n')

    with open(file_path + "train_%d.ltr"%(num),'w',encoding = 'utf-8') as f:
        for item in write_ltr_list:
            f.write(item + '\n')

