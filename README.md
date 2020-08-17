# PJM_Power DataAnalysis
 电力数据预测分析
## 数据来源：
- 美国电力数据PJM https://www.pjm.com/
- 使用的全部已经download to /hourly-energy-consumption/
- AEP
- COMED
- DAYTON
- DEOK
- DUQ
- EKPC
- est
- PJM
- PJME
- PJMW


## 使用框架
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)
- XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way. The same code runs on major distributed environment (Hadoop, SGE, MPI) and can solve problems beyond billions of examples.
## Installation
``` pip install xgboost matplotlib pandas numpy seaborn ```


## Conclusion
- XGBoost 模型较原生的GBDT模型有了较大改进，比较适合序列数据的预测
- 在PJM数据集上的2018-1预测准确率最低，但是其他的时间点基本吻合，原因是2018-12前后有偏离
- 整体趋势基本吻合，以M字型曲线作为最基本趋势图线


## Insights
- 如果前期吻合度较高，有足够理由相信接下来几天的预测较为准确
- 投资方可以看中在电力需求量低的时候价格也较低买入相应的电力股票，相反，则卖出
- 电力公司在预测到需求量降低的时候可以提前做好适当的营销活动以吸引顾客，营销活动切忌不可一直办下去，要分时间段，而需求量的预测结果就是一个很好的判断指标，在需求量较低的时候加大营销投入，吸引用户，相反，则应该停止饥饿营销，因为这是人的天性，人生来喜欢赚小便宜，一旦你的营销活动不再举办的时候，客户感觉不到占了便宜，大概率会纷纷跑路，很难截流，所以及时停止营销也是很好的方法，让客户知道这次活动是真的有期限，过期不候，下一次才会吸引更多客户
- 大的电力公司也可以瞅准时机，预测到电力需求量降低的时候，可以提前准备好收购同行业其他小企业，从而垄断市场
