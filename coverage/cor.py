import pandas as pd
from scipy import stats

print("cifar100-resnet20-size100-cw")

df_coverage = pd.read_csv('corr_data/cifar100-100samples-500groups-kmnc-cw.csv')
print("spearman")
print(df_coverage.corr('spearman'))

p = df_coverage['p'].values.tolist()
errorRate = df_coverage['errorRate'].values.tolist()

# lsc = df_coverage['lsc'].values.tolist()
# dsc = df_coverage['dsc'].values.tolist()
topk = df_coverage['topk'].values.tolist()
nc = df_coverage['nc'].values.tolist()

nbc = df_coverage['nbc'].values.tolist()
snac = df_coverage['snac'].values.tolist()
kmnc = df_coverage['kmnc'].values.tolist()

print("  ")

# corr,p = stats.spearmanr(p,lsc)
# print("probability of adv-lsc", "corr为:", corr, "p值为:", p)
#
# p = df_coverage['p'].values.tolist()
# corr,p = stats.spearmanr(p,dsc)
# print("probability of adv-dsc", "corr为:", corr, "p值为:", p)
#
# print("  ")
#
#
# errorRate = df_coverage['errorRate'].values.tolist()
# corr,p = stats.spearmanr(errorRate,lsc)
# print("error rate-lsc", "corr为:", corr, "p值为:", p)
#
# errorRate = df_coverage['errorRate'].values.tolist()
# corr,p = stats.spearmanr(errorRate,dsc)
# print("error rate-dsc", "corr为:", corr, "p值为:", p)

p = df_coverage['p'].values.tolist()
corr,p = stats.spearmanr(p,topk)
print("probability of adv-topk", "corr为:", corr, "p值为:", p)

p = df_coverage['p'].values.tolist()
corr,p = stats.spearmanr(p,nc)
print("probability of adv-neuron coverage", "corr为:", corr, "p值为:", p)

p = df_coverage['p'].values.tolist()
corr,p = stats.spearmanr(p,nbc)
print("probability of adv-neuron boundary coverage", "corr为:", corr, "p值为:", p)

p = df_coverage['p'].values.tolist()
corr,p = stats.spearmanr(p,snac)
print("probability of adv-strong neuron activation coverage", "corr为:", corr, "p值为:", p)

p = df_coverage['p'].values.tolist()
corr,p = stats.spearmanr(p,kmnc)
print("probability of adv-kmnc", "corr为:", corr, "p值为:", p)
