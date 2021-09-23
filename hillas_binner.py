import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

particle='Proton2'
nobins=5
wbins=10

t1i=np.load('../'+particle+'_t1i.npy')
t2i=np.load('../'+particle+'_t2i.npy')
t3i=np.load('../'+particle+'_t3i.npy')
t4i=np.load('../'+particle+'_t4i.npy')
t1w=np.load('../'+particle+'_t1w.npy')
t2w=np.load('../'+particle+'_t2w.npy')
t3w=np.load('../'+particle+'_t3w.npy')
t4w=np.load('../'+particle+'_t4w.npy')

totalsize=t1i+t2i+t3i+t4i
print(totalsize)
print(np.min(totalsize),np.max(totalsize),np.mean(totalsize),np.std(totalsize))

binned_range=np.linspace(np.min(totalsize),np.max(totalsize),nobins)

fig,axs=plt.subplots(nobins-1,1,figsize=(10,30))
plt.suptitle(particle+' Binned by Size',y=1.0,size='large')

for i in np.arange(nobins-1):
    print(binned_range[i],binned_range[i+1])
    intlocs=np.logical_and(totalsize>binned_range[i],totalsize<binned_range[i+1])
    axs[i].hist(t1w[intlocs],label='T1',alpha=0.1)
    axs[i].hist(t2w[intlocs],label='T2',alpha=0.1)
    axs[i].hist(t3w[intlocs],label='T3',alpha=0.1)
    axs[i].hist(t4w[intlocs],label='T4',alpha=0.1)
    axs[i].set_title(str(binned_range[i])+'<'+str(binned_range[i+1]))
    axs[i].legend(loc='upper right')

plt.tight_layout()
plt.savefig('../Figures/'+particle+'_binnedint.png')
