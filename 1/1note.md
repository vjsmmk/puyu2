InternLM2 技术报告笔记

源社区正在努力缩小专有LLM与开源模型之间的差距。在过去的一年里，如LLaMA ([Touvron et al., 2023a;b](https://arxiv.org/abs/2302.13971))、Qwen ([Bai et al., 2023a](https://arxiv.org/abs/2309.16609))、Mistral ([Jiang et al., 2023](https://arxiv.org/abs/2310.06825))和Deepseek ([Bi et al., 2024](https://arxiv.org/abs/2401.02954))等一些显著的开源大语言模型取得了显著进步。

大语言模型的发展包括预训练、监督微调（SFT）和基于人类反馈的强化学习（RLHF）等主要阶段 ([Ouyang et al., 2022](https://papers.nips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html))。预训练主要基于利用大量的自然文本语料库，积累数万亿的token。这个阶段的目标是为大语言模型配备广泛的知识库和基本技能。预训练阶段的数据质量被认为是最重要的因素。然而，过去关于大语言模型的技术报告 ([Touvron et al., 2023a;b; Bai etal., 2023a; Bi et al., 2024](https://arxiv.org/abs/2401.02954))很少关注预训练数据的处理。InternLM2详细描述了如何为预训练准备文本、代码和长文本数据。

如何有效地延长大语言模型的上下文长度目前是研究的热点，因为许多下游应用，如检索增强生成（RAG） ([Gao et al., 2023](https://arxiv.org/abs/2312.10997))和代理模型 ([Xi et al., 2023](https://arxiv.org/abs/2309.07864))，依赖于长上下文。InternLM2首先采用分组查询注意力（GQA）来在推断长序列时减少内存占用。在预训练阶段，我们首先使用4k个上下文文本训练InternLM2，然后将训练语料库过渡到高质量的32k文本进行进一步训练。最终，通过位置编码外推 ([LocalLLaMA, 2023](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/))，InternLM2在200k个上下文中通过了“大海捞针”测试，表现出色。

1. **开源InternLM2模型展现卓越性能:** 我们已经开源了不同规模的模型包括1.8B、7B和20B，它们在主观和客观评估中都表现出色。此外，我们还发布了不同阶段的模型，以促进社区分析SFT和RLHF训练后的变化。
2. **设计带有200k上下文窗口:** : InternLM2在长序列任务中表现出色，在带有200k上下文的“大海捞针”实验中，几乎完美地识别出所有的“针”。此外，我们提供了所有阶段包括预训练、SFT和RLHF的长文本语言模型的经验。
3. **综合数据准备指导:** 我们详细阐述了为大语言模型（LLM）准备数据的方法，包括预训练数据、特定领域增强数据、监督微调（SFT）和基于人类监督的强化学习（RLHF）数据。这些细节将有助于社区更好地训练LLM。
4. **创新的RLHF训练技术:** 我们引入了条件在线RLHF（COOL RLHF）来调整各种偏好，显著提高了InternLM2在各种主观对话评估中的表现。我们还对RLHF的主观和客观结果进行了初步分析和比较，为社区提供对RLHF的深入理解。

**通信-计算重叠**  

InternEvo进一步通过精心协调通信和计算，以优化整体系统性能。 在使用参数分片时，模型的所有参数分布在多个GPU上以节省GPU内存。在每个训练步 的每个微批次的前向和反向传播过程中，InternEvo会高效地预加载即将到来的层的完整参数集，同时计算当前层。生成的梯度在参数分片组内通过ReduceScatter进行同步，然后通过AllReduce跨参数分片组同步。这些通信过程巧妙地与反向计算重叠，最大化训练管道的效率。对于优化器状态分片，当GPU在分片组内通过Broadcast广播更新的参数时， InternEvo会与下一个训练步骤的前向计算进行战略重叠。这些创新的重叠方法有效地平衡了通信开销和计算执行时间，显著提高了整体系统性能。

**长序列训练**  

长序列训练的主要挑战之一是计算速度和通信开销之间的权衡。InternEvo将GPU内存管理分解为四个并行维度（数据、 张量、 序列和管道）和三个分片维度（参数、梯度和优化器状态）([Chen et al., 2024a](https://arxiv.org/abs/2401.09149))。我们对每个维度的内存和通信成本进行了详尽分析，并使用执行模拟器来识别和实施最优的并行化策略。根据训练规 模、序列长度、模型大小和批量大小，可以自动搜索最优执行计划。通过这种执行计划， InternEvo能够处理长达100万个令牌的长序列训练。此外，InternEvo还实施了内存管理技术来减少GPU内存碎片，这是长序列训练场景中的常见问题。它使用内存池进行统一内存管理，并引入了一种碎片整理技术，以主动合并小内存块，防止内存不足错误。

## 2.2  模型结构

Transformer（[Vaswani et al. (2017)](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)）由于其出色的并行化能力，已经成为过去大语言模型（LLMs）的主流选择，这充分利用了GPU的威力（[Brown et al. (2020); Chowdhery et al. (2023); Zeng et al. (2023)](https://jmlr.org/papers/v24/22-1144.html)）。LLaMA（[Touvron et al. (2023a)](https://arxiv.org/abs/2302.13971)）在Transformer架构基础上进行了改进，将LayerNorm（[Ba et al. (2016)](https://arxiv.org/abs/1607.06450)）替换为RMSNorm（[Zhang & Sennrich (2019)](https://proceedings.neurips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html)），并采用SwiGLU（[Shazeer (2020)](https://arxiv.org/abs/2002.05202)）作为激活函数，从而提高了训练效率和性能。  自从LLaMA（[Touvron et al. (2023a)](https://arxiv.org/abs/2302.13971)）发布以来，社区积极地扩展了基于LLaMA架构的生态系统，包括高效推理的提升（[lla (2023)](https://github.com/ggerganov/llama.cpp)）和运算符优化（[Dao (2023)](https://arxiv.org/abs/2307.08691)）等。为了确保我们的模型InternLM2能无缝融入这个成熟的生态系统，与Falcon（[Almazrouei et al. (2023)](https://arxiv.org/abs/2311.16867)）、Qwen（[Bai et al. (2023a)](https://arxiv.org/abs/2309.16609)）、Baichuan（[Yang et al. (2023)](https://arxiv.org/abs/2309.10305)）、Mistral（[Jiang et al. (2023)](https://arxiv.org/abs/2310.06825)）等知名LLMs保持一致，我们选择遵循LLaMA的结构设计原则。为了提高效率，我们将$$W_k$$、$$W_q$$和$$W_v$$矩阵合并，这在预训练阶段带来了超过5%的训练加速。此外，为了更好地支持多样化的张量并行（tp）变换，我们重新配置了矩阵布局。对于每个head的$$W_k$$、$$W_q$$和$$W_v$$，我们采用了交错的方式

## 3.1 预训练数据

大规模语言模型（LLM）的预训练深受数据的影响，数据加工主要面对的挑战包含敏感数据的处理、全面知识的覆盖以及效率与质量的平衡。在本节中将介绍我们在通用领域的文本数据、编程语言相关数据和长文本数据的处理流程。

### 3.1.1 文本数据

我们的预训练数据集的来源为网页、论文、专利和书籍。为了将这些原始数据转化为预训练数据集，我们首先将所有数据标准化为指定格式，然后根据内容类型和语言进行分类，并将结果存储为JSON Lines（jsonl）格式；然后，对所有数据，我们应用了包括基于规则的过滤、数据去重、安全过滤和质量过滤等多个处理步骤。这使得我们得到了一个丰富、安全且高质量的文本数据集。

**数据处理****流程**

本工作中使用的数据处理流程如图[3](https://aicarrier.feishu.cn/wiki/Xarqw88ZkimmDXkdTwBcuxEfnHe#YRKQdlAPsokfLGxbDNAcaRTDndb)所示。整个数据处理流程首先将来自不同来源的数据标准化以获得**格式化数据**。然后，使用启发式统计规则对数据进行过滤以获得**干净****数据**。接下来，使用局部敏感哈希（LSH）方法对数据去重以获得**去重数据**。然后，我们应用一个复合安全策略对数据进行过滤，得到**安全数据**。我们对不同来源的数据采用了不同的质量过滤策略，最终获得**高质量预训练数据**。

**数据格式化**

我们将以网页数据为例详细介绍数据处理流程。我们的网页数据主要来自[Common Crawl](https://commoncrawl.org/)。首先，我们需要解压缩原始的Warc格式文件，并使用Trafilatura ([Barbaresi, 2021](https://arxiv.org/html/2403.17297v1#bib.bib13))进行HTML解析和主文本提取。然后，我们使用[pycld2](https://pypi.org/project/pycld2/)库进行语言检测和主文本分类。最后，我们为数据分配一个唯一标识符，并以jsonl（JSON行）格式存储，从而获得**格式化数据**。

**基于规则的****处理**

从互联网随机提取的网页数据通常包含大量低质量数据，如解析错误、格式错误和非自然语言文本。常见的做法是设计基于规则的正则化和过滤方法来修改和过滤数据，如 ([Rae et al., 2021](https://arxiv.org/html/2403.17297v1#bib.bib70))、C4 ([Dodge et al., 2021](https://arxiv.org/html/2403.17297v1#bib.bib33))和RefinedWeb ([Penedo et al., 2023](https://arxiv.org/html/2403.17297v1#bib.bib67))。基于对数据的观察，我们设计了一系列启发式过滤规则，重点关注分隔和换行中的异常、异常字符的频率以及标点符号的分布。通过应用这些过滤器，我们得到了**干净数据**。

**去重**

互联网上存在的大量重复文本会对模型训练产生负面影响。因此，我们采用基于Locality-Sensitive Hashing (LSH)的方法对数据进行模糊去重。更具体地说，我们使用MinHash方法([Broder, 1997](https://arxiv.org/html/2403.17297v1#bib.bib16))，在文档的5-gram上使用128个哈希函数建立签名，并使用0.7作为去重阈值。我们的目标是保留最新数据，即优先考虑具有较大Common Crawl数据集版本号的数据。在LSH去重后，我们得到了**去重数据**。

**安全过滤**

互联网上充斥着有毒和色情的内容，使用这些内容进行模型训练会对模型的表现产生负面影响，增加生成不安全内容的可能性。因此，我们采用了一种综合性的安全策略，结合了“域名屏蔽”、“关键词屏蔽”、“色情内容分类器”和“有害性分类器”来过滤数据。具体来说，我们构建了一个包含大约1300万个不安全域名的屏蔽域名列表，以及一个包含36,289个不安全词汇的屏蔽词列表，用于初步的数据过滤。考虑到关键词屏蔽可能会无意中排除大量数据，我们在编制屏蔽词列表时采取了谨慎的方法。

为了进一步提高不安全内容的检测率，我们使用了来自Kaggle的“有害评论分类挑战赛（Toxic Comment Classification Challenge）”数据集对BERT模型进行了微调，从而得到了一个有害性分类器。我们从去重后的数据中抽取了一些样本，并使用[Perspective API](https://perspectiveapi.com/)对其进行了标注来创建色情分类数据集然后，我们用这个数据集微调BERT模型，产生一个色情分类器。最后，通过使用这两个分类器对数据进行二次过滤，过滤掉分数低于阈值的数据，我们得到了**安全数据**。

预训练阶段为大型语言模型（LLMs）赋予了解决各种任务所需的基础能力和知识。我们进一步微调LLMs，以充分激发其能力，并指导LLMs作为有益和无害的AI助手。这一阶段，也常被称为“对齐”（Alignment），通常包含两个阶段：监督微调（SFT）和基于人类反馈的强化学习（RLHF）。在SFT阶段，我们通过高质量指令数据（见[4.1 监督微调](https://aicarrier.feishu.cn/wiki/Xarqw88ZkimmDXkdTwBcuxEfnHe?fromScene=spaceOverview#Ob2CdIm7OoDk32xPGMWc2RB3nob)）微调模型，使其遵循多种人类指令。然后我们提出了带人类反馈的条件在线强化学习（**CO**nditional**O**n**L**ine Reinforcement Learning with Human Feedback，COOL RLHF），它应用了一种新颖的条件奖励模型，可以调和不同的人类偏好（例如，多步推理准确性、有益性、无害性），并进行三轮在线RLHF以减少奖励黑客攻击（见[4.2 基于人类反馈的条件在线强化学习COOL RLHF](https://aicarrier.feishu.cn/wiki/Xarqw88ZkimmDXkdTwBcuxEfnHe?fromScene=spaceOverview#YqzAdDoRtolGKKxY9LDcAAqqnlZ)）。在对齐阶段，我们通过在SFT和RLHF阶段利用长上下文预训练数据来保持LLMs的长上下文能力（见[4.3 长文本微调](https://aicarrier.feishu.cn/wiki/Xarqw88ZkimmDXkdTwBcuxEfnHe?fromScene=spaceOverview#AnkzdSPAyoWPDBx4vrZcuyIpnHb)）。我们还介绍了我们提升LLMs工具利用能力的实践（参见[4.4 工具增强的LLMs](https://aicarrier.feishu.cn/wiki/Xarqw88ZkimmDXkdTwBcuxEfnHe?fromScene=spaceOverview#PkIIdTeJYoQeeKx1BIrcUrXUnPh)）。

基于人类反馈的强化学习（RLHF）(Christiano et al., 2017; Ouyang et al., 2022) 是大型语言模型领域内的一种创新方法。通过融入人类反馈，RLHF创建了了代理人类偏好的奖励模型，从而通过使用近端策略优化（Proximal Policy Optimization, PPO）(Schulman et al., 2017) 为大型语言模型（LLM）提供用于学习的奖励信号。这种方法使得模型能更好地理解和执行难以通过传统方法定义的任务。

尽管RLHF取得了成就，但其实际应用中仍存在一些问题。首先是偏好冲突。例如，在开发对话系统时，我们期望它提供有用的信息（有益）的同时不产生有害或不适当的内容（无害）。然而，在实际中，这两者往往无法同时满足，因为提供有用的信息在某些情况下可能涉及敏感或高风险内容。现有的RLHF方法 (Touvron et al., 2023b; Dai et al., 2023; Wu et al., 2023) 通常依赖于多个偏好模型进行评分，这也使得训练管道中引入了更多的模型，从而增加了计算成本并减慢了训练速度。其次，RLHF面临奖励滥用（reward hacking）的问题，特别是当模型规模增大，策略变得更强大时 (Manheim & Garrabrant, 2018; Gao et al., 2022)，模型可能会通过捷径“欺骗”奖励系统以获得高分，而不是真正学习期望的行为。这导致模型以非预期的方式最大化奖励，严重影响LLMs的有效性和可靠性。

为了解决这些问题，我们提出了条件在线RLHF（COOL RLHF）。COOL RLHF首先引入了一个条件奖励机制来调和不同的偏好，允许奖励模型根据特定条件动态地分配其注意力到各种偏好上，从而最优地整合多个偏好。此外，COOL RLHF采用多轮在线RLHF策略，以使LLM能够快速适应新的人类反馈，减少奖励滥用的发生

**数据组成** 条件奖励模型的训练过程涉及一个庞大的数据集，包括对话、文章写作、诗歌、总结、编程、数学和格式化输出等各种领域，共有高达240万二进制偏好对。这个全面的数据集确保了模型的广泛适应性，并增强了其在更广泛、更复杂场景下进行强化学习的能力。因此，通过使用条件系统提示方法，奖励模型可以响应复杂的人类需求，在PPO阶段提供对奖励分数的更精细控制。

**损失函数** 此外，为了减少数据集中简单样本和困难样本之间不平衡的影响，受到Focal Loss(Lin et al., 2017)的启发，我们修改了原始排名损失函数(Burges et al., 2005)。我们在排名损失中添加了一个难度衰减系数，使得困难样本的损失值更大，简单样本的损失值更小，以防止过拟合量简单样本。聚焦排名损失（Focal Ranking Loss）的公式为

在强化学习对齐阶段，我们采用了标准的近端策略优化（Proximal Policy Optimization，PPO）算法，并对它做了一些适应性修改，以确保训练过程更加稳定。该框架包括四个模型：演员模型、评论家模型、参考模型和奖励模型。在训练过程中，后两个模型被冻结，只有前两个模型是主动训练的。值得注意的是，所有这些模型的尺寸都相同，确保了它们处理和生成数据的能力一致。我们在大约400个迭代中遍历了约20万个多样化的查询，并选择在验证集上的最佳检查点进行发布。

## 4.3 长文本微调

为了在微调和RLHF之后保留LLM的长上下文能力，我们受到了之前采用长上下文预训练语料库的SFT工作(Xiong et al., 2023)的启发，在SFT和RLHF中继续使用长上下文预训练数据。具体来说，我们使用了两种数据：一种是从书籍中获取的长上下文数据，另一种是从GitHub仓库中获得并通过特定范式连接的长上下文数据，具体如下所述。

为了增强InternLM2的数据分析能力，我们选择了在DS-1000(Lai et al., 2023)中使用的代码仓库作为核心仓库，包括Pandas、Numpy、Tensorflow、Scipy、Scikit-learn、PyTorch和Matplotlib。然后我们在GitHub上搜索了引用这些核心仓库并拥有超过10,000个星标的开源仓库，并对这些仓库进行了与预训练相同的过滤和数据清洗过程。 对于每个仓库，我们首先使用深度优先方法对获得的原始数据进行排序，同时生成所需的提示，简要描述文件内容，如图[11](https://aicarrier.feishu.cn/wiki/Xarqw88ZkimmDXkdTwBcuxEfnHe#IWYLdeBiTorO8Wx1TVhcQQi2nhJ)所示。随后，我们将处理后的数据按顺序连接，直到达到32k的长度。实验结果表明，长上下文代码数据不仅提高了LLM的长上下文能力，也提高了代码能力。