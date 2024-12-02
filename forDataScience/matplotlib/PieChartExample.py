import pandas as pd
import matplotlib.pyplot as plt

years=[1950,1955,1960,1965,1970,1980,1985,1990,1995,2000,2005,2010,2015]
pops=[2.5,2.7,3.0,3.3,3.6,4.0,4.4,4.8,5.3,5.7,6.1,6.5,7.3]
death=[1.2,1.1,1.2,2.1,2.0,2.3,1.8,1.9,2.6,1.6,2.4,2.4,4.0]

data = {'Year': years, 'Population (Billions)': pops, 'Deaths (Millions)': death}
df = pd.DataFrame(data)
plt.figure(figsize=(8, 8))
plt.pie(df['Deaths (Millions)'], labels=df['Year'], autopct='%1.1f%%', startangle=140)

# Chart title
plt.title('Proportion of Deaths (Millions) Over the Years')
plt.show()