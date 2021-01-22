import numpy as np
import pandas as pd

submission = pd.read_csv('./practice/dacon/data/sample_submission.csv')
quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in range(2):
    print(f'Day {i} 진행중...')
    # x_train, x_val, y1_train, y1_val, y2_train, y2_val = tts(x[i],y1[i],y2[i], train_size = 0.7,shuffle = True, random_state = 0)
    # 내일!
    for j in quantiles:
        x = []
        for k in range(5):
            pred = np.array([k])
            x.append(pred)
        x = pd.DataFrame(x,columns = [f'q_{j:.1f}'])
        df_temp1 = pd.concat(x, axis = 0)
        df_temp1[df_temp1<0] = 0
        num_temp1 = df_temp1.to_numpy()
        submission.loc[submission.id.str.contains(f"Day7_{int(i/2)}h{(i%2)*30}m"), ["q_{j:.1f}"]] = num_temp1

print(submission)