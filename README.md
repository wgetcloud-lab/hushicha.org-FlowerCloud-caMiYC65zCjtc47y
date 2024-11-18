[合集 \- 机器学习(12\)](https://github.com)[1\.从零开始学机器学习——什么是机器学习09\-24](https://github.com/guoxiaoyu/p/18412875)[2\.从零开始学机器学习——了解回归09\-25](https://github.com/guoxiaoyu/p/18413894)[3\.从零开始学机器学习——准备和可视化数据09\-27](https://github.com/guoxiaoyu/p/18419035)[4\.从零开始学机器学习——线性和多项式回归09\-29](https://github.com/guoxiaoyu/p/18421693)[5\.从零开始学机器学习——逻辑回归09\-30](https://github.com/guoxiaoyu/p/18429831)[6\.从零开始学机器学习——网络应用10\-06](https://github.com/guoxiaoyu/p/18438177):[veee加速器](https://liuyunzhuge.com)[7\.从零开始学机器学习——了解分类算法10\-14](https://github.com/guoxiaoyu/p/18444408)[8\.从零开始学机器学习——初探分类器10\-15](https://github.com/guoxiaoyu/p/18445932)[9\.从零开始学机器学习——分类器详解10\-16](https://github.com/guoxiaoyu/p/18446455)[10\.从零开始学机器学习——构建一个推荐web应用10\-17](https://github.com/guoxiaoyu/p/18447175)[11\.从零开始学机器学习——了解聚类11\-17](https://github.com/guoxiaoyu/p/18541006)12\.从零开始学机器学习——聚类可视化11\-18收起
首先给大家介绍一个很好用的学习地址：[https://cloudstudio.net/columns](https://github.com)


在上一章节中，我们对聚类的相关知识进行了全面的介绍，旨在为大家打下坚实的理论基础。今天，我们的主要任务是深入探讨数据可视化的技术和方法。在之前的学习中，我们已经接触过回归分析中的可视化技术，而今天我们将专注于聚类分析的可视化。我们将学习如何使用散点图、同心圆等可视化工具，以更直观地理解聚类结果。


# 数据可视化——聚类


今天我们的目标是从一个特定的文件中读取和分析数据。该文件包含了大量的歌曲信息，涵盖多个字段，例如歌曲名称、音乐流派、歌唱家、流行度、可舞性、发布时间等。在我们的分析过程中，我们将首先过滤出这份数据中最为突出的三大流派，并提取相关数据。接下来，我们将深入探讨这三大流派在其他字段上的相关性，并分析其数据分布情况。


需要注意的是，本章节并不打算过多讨论聚类算法及其具体作用，我们的重点将放在如何运用可视化工具来展示和理解这些数据。这将有助于我们更直观地捕捉到数据中的趋势和模式，从而为后续的分析打下基础。


## 过滤数据


首先，我们需要引入一些关键的依赖包：



> !pip install seaborn



```
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("../data/nigerian-songs.csv")
df.head()

```

接下来，我们将对数据集进行初步查看，以了解其整体结构和内容。


![image](https://img2024.cnblogs.com/blog/1423484/202411/1423484-20241112081341680-882588454.png)


使用以下命令，我们可以全面查看数据的大致格式以及数据量等关键信息。



```
df.info()
df.isnull().sum()
df.describe()

```

df.info()：快速了解数据的结构和列的类型。


df.isnull().sum()：识别哪些列存在缺失数据以及缺失的程度。


df.describe()：主要用于数值型数据，提供了每列的基本统计特性，便于理解数据的分布情况。


我们可以先查看一下describe方法输出的数据，这部分信息将为我们提供重要的统计结果和数据分布情况。其他相关的内容我们之前已经讨论过，具体情况可以参考附图。


![image](https://img2024.cnblogs.com/blog/1423484/202411/1423484-20241112081347102-1381384783.png)


### 数据筛选


接下来，我们将对数据进行筛选，目标是提取出最流行的三大音乐流派。为了实现这一目标，我们将以artist\_top\_genre作为X轴，以便更清晰地观察数据的分布情况。以下是相应的代码：



```
import seaborn as sns

top = df['artist_top_genre'].value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=top[:5].index,y=top[:5].values)
plt.xticks(rotation=45)
plt.title('Top genres',color = 'blue')

```

如图所示，我们提取出了前五个音乐流派，并成功识别出其中的三个：afro dancehall、afropop以及nigerian pop。


![image](https://img2024.cnblogs.com/blog/1423484/202411/1423484-20241112081353543-2060073141.png)


请注意，由于在检查数据时未发现任何缺失值（即没有null数据），因此我们决定不删除任何行，直接进行绘图。然而，如果你的数据集中存在缺失值，建议你在进行绘图之前，首先删除包含缺失值的行，以确保数据的完整性和图形的准确性。这样可以避免潜在的数据偏差，确保分析结果的可靠性。



```
df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
df = df[(df['popularity'] > 0)]
top = df['artist_top_genre'].value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=top.index,y=top.values)
plt.xticks(rotation=45)
plt.title('Top genres',color = 'blue')

```

我们的数据筛选工作终于圆满完成。现在，我们已经识别出当前最受欢迎的三大流派，具体信息如图所示。


![image](https://img2024.cnblogs.com/blog/1423484/202411/1423484-20241112081359386-1585689347.png)


### 强相关性


同样地，让我们再来查看一下热力图。这一部分内容我们在回归分析中已经详细讲解过，因此在这里我们将直接提供相关的代码。以下是具体的实现代码：



```
corrmat = df.corr(numeric_only=True)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)

```

根据图片所示的数据分析，我们可以清楚地看到，唯一表现出强相关性的变量是能量（energy）与响度（loudness）之间的关系。这一点并不令人惊讶，因为嘈杂的音乐往往伴随着极高的活力和强烈的节奏感。


![image](https://img2024.cnblogs.com/blog/1423484/202411/1423484-20241112081405592-1139242046.png)


接下来，我们将深入探讨一种新的可视化方法，以帮助我们更好地理解聚类分析中的数据分布情况。


## 数据分布


### 同心圆


接下来，我们将根据受欢迎程度和可舞性这两个指标进行数据分析，具体方式包括绘制同心圆和散点图。这些图表将帮助我们更直观地理解数据的分布和趋势。当然，你也可以选择其他字段进行对比分析，完全可以根据个人的喜好和需求进行调整。



```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.iloc[:, 6:8] = df.iloc[:, 6:8].apply(LabelEncoder().fit_transform)

sns.set_theme(style="ticks") 
g = sns.jointplot(
    data=df,
    x="popularity", y="danceability", hue="artist_top_genre",
    kind="kde",
)

```

由于数据分布和数据类型不一致，为了确保分析的准确性和一致性，我决定将所有数据统一转换为整数格式。如图所示：


![image](https://img2024.cnblogs.com/blog/1423484/202411/1423484-20241112081413211-20511015.png)


他的目的是成一个联合分布图，用于展示数据集中流行度（popularity）和舞蹈性（danceability）之间的关系，同时通过不同颜色标识不同的音乐风格（artist\_top\_genre）


### 散点图



```
sns.FacetGrid(df, hue="artist_top_genre").map(plt.scatter, "popularity", "danceability",s=5) .add_legend()

```

一行代码即可观察其散点分布，如图所示：


![image](https://img2024.cnblogs.com/blog/1423484/202411/1423484-20241112081418685-948130613.png)


一般来说，对于聚类分析，使用散点图来展示数据的聚类效果是非常有效的，因此掌握这种可视化技术对我们理解数据的结构和模式至关重要。在接下来的课程中，我们将利用经过过滤后的数据，采用 k\-means 聚类算法来探索和识别数据中以有趣方式重叠的组。


# 总结


在本章节中，我们深入探讨了数据可视化在聚类分析中的应用。通过对歌曲信息数据集的分析，我们成功识别了三大流派，并运用散点图和同心圆等可视化工具，直观地展示了数据的分布与趋势。可视化不仅增强了我们对数据的理解，还为后续的聚类分析打下了坚实的基础。


通过这种方式，我们不仅能识别出数据中的模式，还能为决策提供有力支持。正如我们所见，数据的可视化过程是一个探索性的旅程，它帮助我们在复杂的数据中找到隐藏的联系和意义。接下来，我们将应用 k\-means 聚类算法，进一步挖掘这些数据背后的故事。




---


我是努力的小雨，一名 Java 服务端码农，潜心研究着 AI 技术的奥秘。我热爱技术交流与分享，对开源社区充满热情。同时也是一位腾讯云创作之星、阿里云专家博主、华为云云享专家、掘金优秀作者。


💡 我将不吝分享我在技术道路上的个人探索与经验，希望能为你的学习与成长带来一些启发与帮助。


🌟 欢迎关注努力的小雨！🌟


