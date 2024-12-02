import matplotlib.pyplot as plt
import pandas as pd

years=[1950,1955,1960,1965,1970,1980,1985,1990,1995,2000,2005,2010,2015]
pops=[2.5,2.7,3.0,3.3,3.6,4.0,4.4,4.8,5.3,5.7,6.1,6.5,7.3]
death=[1.2,1.1,1.2,2.1,2.0,2.3,1.8,1.9,2.6,1.6,2.4,2.4,4.0]
# plt.plot(years,pops,color=(255/255,100/255,100/255))
# plt.plot(years,death,color=(.6,.6,1))
# plt.ylabel("Population in Billions")
# plt.xlabel("Population growth by year")
# plt.title("Population Growth")
# plt.legend()
# plt.show()
# plt.close()


lines=plt.plot(years,pops,years,death)
plt.grid(True)
plt.setp(lines,color=(1,.4,.5),marker='*')
plt.show()


