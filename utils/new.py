import torchfile

csv_list = []

if __name__ == '__main__':
  ffile = open('lua_filenames.txt')
  # ffile = open('railway.txt')
  while 1:
    audio_file = ffile.readline()
    if not audio_file:
       break
    # import pdb; pdb.set_trace()
#     print(audio_file)
    
    f = torchfile.load(audio_file[0:-1])
#     import pdb; pdb.set_trace()
#     print(f[b'feature'].shape)
    # sys.exit()
#     pdb.set_trace()
    res = {}
    res['name'] = audio_file[0:-1]
#     res['feature'] = f[b'feature']
    res['feature'] = list(f[b'feature'].reshape((1*256*27*1,)))
    csv_list.append(res)    

    # import pdb; pdb.set_trace()

#   pdb.set_trace()

  import pandas as pd
  tmp=pd.DataFrame.from_records(csv_list)
  tmp.to_csv('new.csv')
