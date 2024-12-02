import numpy as np
import pandas as pd

my_data=np.random.randint(0,100, (4, 3))
countries=['Poland', 'Germany', 'Czech Republic', 'Lithuania']
column=['jan', 'feb', 'mar']

df=pd.DataFrame(my_data, index=countries, columns=column)
print(df)